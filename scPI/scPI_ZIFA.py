import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from torch import optim


def weight_init(m, std=0.01):
	if isinstance(m, nn.Linear):
		m.weight.data.normal_(0, std)
		m.bias.data.fill_(0)


class Encoder(nn.Module):
	def __init__(self, n_input, n_latent, non_linear=True):
		super(Encoder, self).__init__()
		self.n_input = n_input
		self.n_latent = n_latent
		self.non_linear = non_linear

		# layers for encoder
		if self.non_linear:
			self.fc = nn.Sequential(
				nn.Linear(self.n_input, 1024),
				nn.ReLU(),
				nn.Linear(1024, self.n_input),
				nn.ReLU()
			)
			self.fc.apply(weight_init)
		self.e_fc_z1 = nn.Linear(self.n_input, self.n_latent)
		self.e_fc_z2 = nn.Linear(self.n_input, self.n_latent)
		self.e_fc_z1.apply(weight_init)
		self.e_fc_z2.apply(weight_init)

	def sample_latent(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		return mu + std * eps

	def forward(self, y):
		if self.non_linear:
			y = self.fc(y)
		mu_z = self.e_fc_z1(y)
		logvar_z = self.e_fc_z2(y)
		z_sample = self.sample_latent(mu_z, logvar_z)
		return mu_z, logvar_z, z_sample


class Decoder(nn.Module):
	def __init__(self, n_input: int, n_latent: int, act_nng_exp=True):
		super(Decoder, self).__init__()
		self.n_input = n_input
		self.n_latent = n_latent
		self.act_nng_exp = act_nng_exp

		self.loglam = nn.Parameter(torch.randn(1))
		self.logW = nn.Parameter(torch.randn(self.n_input))

		# layers for decoder
		self.W_netp = nn.Parameter(torch.Tensor(self.n_input, self.n_latent).normal_(mean=0.0, std=0.01))
		self.b_netp = nn.Parameter(torch.Tensor(self.n_input).normal_(mean=0.0, std=0.01))

	def forward(self, z):
		x_tilde = F.linear(z, self.W_netp, self.b_netp)
		if self.act_nng_exp:
			lam = torch.exp(self.loglam)
			W   = torch.exp(self.logW)
		else:
			lam = F.softplus(self.loglam)
			W   = F.softplus(self.logW)
		return x_tilde, lam, W


def elbo(y, n_input, n_latent, z_sample, x_tilde, W, lam, qz_logvar, eps=1e-6):
	# elbo_logcll_z       = torch.sum(- 0.5*torch.sum(z_sample**2, axis=1) - n_latent/2 * np.log(2*np.pi))
	elbo_logcll_z       = torch.sum(- 0.5*(z_sample**2) - 1/2 * np.log(2*np.pi))
	elbo_logcll_zero    = torch.sum((-(x_tilde**2)*lam/(1+2*W*lam) - 0.5*torch.log(1+2*W*lam+eps)) * (y<eps))
	elbo_logcll_nonzero = torch.sum((-(y-x_tilde)**2/2/W - 0.5*np.log(2*np.pi) - 0.5*torch.log(W+eps) + torch.log(1-torch.exp(-lam*(y**2))+eps)) * (y>eps))
	elbo_entropy_z      = torch.sum(0.5*np.log(2*np.pi) + 0.5*qz_logvar + 0.5)
	Q                   = elbo_logcll_z + elbo_logcll_zero + elbo_logcll_nonzero
	elbo                = Q + elbo_entropy_z
	return elbo, Q


def fitModel(Y, K, batch_size=128, n_epoch=500, lr=5e-4, act_nng_exp=True, non_linear=True, Y_test=None, eval_train=True):

	t0 = time.time()

	# basic setup
	torch.set_default_dtype(torch.float32)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	N, D = Y.shape
	if (batch_size == N):
		Y_train_np = Y
		Y_train = torch.from_numpy(np.array(Y_train_np)).to(device)
	iterep = int(N / float(batch_size))

	# test setup
	if (not Y_test is None):
		N_test, _ = Y_test.shape
		Y_test_gpu = torch.from_numpy(np.array(Y_test)).to(device)

	# define nets and optimizer
	Enet = Encoder(D, K, non_linear=non_linear).to(device)
	Dnet = Decoder(D, K, act_nng_exp=act_nng_exp).to(device)
	Eoptimizer = optim.Adam(Enet.parameters(), lr=lr)
	Doptimizer = optim.Adam(Dnet.parameters(), lr=lr)

	epoches = []
	losses  = []
	logllhs = []
	times   = []
	losses_test = []

	# train
	for t in range(iterep * n_epoch):

		if (batch_size < N):
			index_train = np.random.choice(np.arange(N), size=batch_size)
			Y_train_np = Y[index_train, :]
			Y_train = torch.from_numpy(Y_train_np).to(device)

		# encoder and decoder
		Enet.train()
		Dnet.train()
		Enet.zero_grad()
		Dnet.zero_grad()
		qz_mu, qz_logvar, z_sample = Enet(Y_train)
		x_tilde, lam, W = Dnet(z_sample)
		elbo_v, Q = elbo(Y_train, D, K, z_sample, x_tilde, W, lam, qz_logvar)
		loss = - elbo_v / batch_size + np.mean(np.sum(Y_train_np, axis=-1))
		logllh_v = Q / N - np.mean(np.sum(Y_train_np, axis=-1))
		loss.backward()
		Eoptimizer.step()
		Doptimizer.step()

		# record
		if (t+1) % iterep == 0:
			# print(int((t+1)/iterep), loss.cpu().data.numpy())
			epoches.append((t+1)/iterep)
			losses.append(loss.item())
			logllhs.append(logllh_v.item())
			times.append(time.time() - t0)
			if (not Y_test is None):
				with torch.no_grad():
					qz_mu, qz_logvar, z_sample = Enet(Y_test_gpu)
					x_tilde, lam, W = Dnet(z_sample)
					elbo_v, Q = elbo(Y_test_gpu, D, K, z_sample, x_tilde, W, lam, qz_logvar)
					loss_test = - elbo_v / N_test + np.mean(np.sum(Y_test, axis=-1))
					losses_test.append(loss_test.item())

	# evaluate training (use whole dataset)
	result = {}
	if eval_train:
		with torch.no_grad():
			Y_train = torch.from_numpy(np.array(Y)).to(device)

			Enet.eval()
			Dnet.eval()
			qz_mu, qz_logvar, z_sample = Enet(Y_train)
			x_tilde, lam, W = Dnet(z_sample)
			elbo_v, Q = elbo(Y_train, D, K, z_sample, x_tilde, W, lam, qz_logvar)
			loss = - elbo_v / N + np.mean(np.sum(Y, axis=-1))
			logllh_v = Q / N - np.mean(np.sum(Y, axis=-1))

			A_c   = np.array(Dnet.W_netp.clone().detach().cpu().numpy())
			mu_c  = np.array(Dnet.b_netp.clone().detach().cpu().numpy()).reshape([D, 1])
			W_c   = np.array(W.clone().detach().cpu().numpy()).reshape([D, 1])
			lam_c = np.array(lam.clone().detach().cpu().numpy())[0]

			result["logllh"] = logllh_v.item()
			result["loss"] = loss.item()
			result["latent"] = z_sample.cpu().numpy()
			result["x_tilde"] = x_tilde.cpu().numpy()
			result["lam"] = lam_c
			result["W"] = W_c
			result["A"] = A_c
			result["mu"] = mu_c
	result["epochs"] = epoches
	result["losses"] = losses
	result["logllhs"] = logllhs
	result["times"] = times

	# evaluate testing (use whole dataset)
	result_test = {}
	if (not Y_test is None):
		with torch.no_grad():
			Enet.eval()
			Dnet.eval()
			qz_mu, qz_logvar, z_sample = Enet(Y_test_gpu)
			x_tilde, lam, W = Dnet(z_sample)
			elbo_v, Q = elbo(Y_test_gpu, D, K, z_sample, x_tilde, W, lam, qz_logvar)
			loss   = - elbo_v / N_test + np.mean(np.sum(Y_test, axis=-1))
			logllh_v = Q / N_test - np.mean(np.sum(Y_test, axis=-1))

			A_c   = np.array(Dnet.W_netp.clone().detach().cpu().numpy())
			mu_c  = np.array(Dnet.b_netp.clone().detach().cpu().numpy()).reshape([D, 1])
			W_c   = np.array(W.clone().detach().cpu().numpy()).reshape([D, 1])
			lam_c = np.array(lam.clone().detach().cpu().numpy())[0]

			result_test["logllh"] = logllh_v.item()
			result_test["loss"] = loss.item()
			result_test["latent"] = z_sample.detach().cpu().numpy()
			result_test["x_tilde"] = x_tilde.detach().cpu().numpy()
	result_test["losses"] = losses_test

	return result, result_test

