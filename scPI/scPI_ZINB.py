import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import sys
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

		# layers for encoder of z
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

		# layers for encoder of l
		self.enl = nn.Sequential(
				nn.Dropout(0.1),
				nn.Linear(self.n_input, 128),
				nn.BatchNorm1d(128),
				nn.ReLU(),
				nn.Dropout(0.1),
				nn.Linear(128, 128),
				nn.BatchNorm1d(128),
				nn.ReLU()
			)
		self.enl.apply(weight_init)
		self.e_fc_l1 = nn.Linear(128, 1)
		self.e_fc_l2 = nn.Linear(128, 1)
		self.e_fc_l1.apply(weight_init)
		self.e_fc_l2.apply(weight_init)

	def sample_latent(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		return mu + std * eps

	def forward(self, y):
		# for z
		h = torch.log(y + 1)
		if self.non_linear:
			h = self.fc(h)
		mu_z = self.e_fc_z1(h)
		logvar_z = self.e_fc_z2(h)
		z_sample = self.sample_latent(mu_z, logvar_z)

		# for l
		w = torch.log(y + 1)
		w = self.enl(w)
		mu_l = self.e_fc_l1(w)
		logvar_l = self.e_fc_l2(w)
		l_sample = self.sample_latent(mu_l, logvar_l)
		return mu_z, logvar_z, z_sample, mu_l, logvar_l, l_sample


class Decoder(nn.Module):
	def __init__(self, n_input: int, n_latent: int, act_nn_exp):
		super(Decoder, self).__init__()
		self.n_input = n_input
		self.n_latent = n_latent
		self.act_nn_exp = act_nn_exp

		self.logtheta = nn.Parameter(torch.randn(self.n_input))

		# layers for decoder
		self.W1_netp = nn.Parameter(torch.Tensor(self.n_input, self.n_latent).normal_(mean=0.0, std=0.01))
		self.b1_netp = nn.Parameter(torch.Tensor(self.n_input).normal_(mean=0.0, std=0.01))
		self.W2_netp = nn.Parameter(torch.Tensor(self.n_input, self.n_latent).normal_(mean=0.0, std=0.01))
		self.b2_netp = nn.Parameter(torch.Tensor(self.n_input).normal_(mean=0.0, std=0.01))
		self.smx = nn.Softmax(dim=1)

	def forward(self, z, l):
		scale_rho = F.linear(z, self.W1_netp, self.b1_netp)
		scale_rho = self.smx(scale_rho)
		rho       = torch.exp(l.view([-1, 1])) * scale_rho

		logitpi   = F.linear(z, self.W2_netp, self.b2_netp)
		
		if self.act_nn_exp:
			theta = torch.exp(self.logtheta)
		else:
			theta = F.softplus(self.logtheta)
		return theta, rho, logitpi


def elbo(y, n_input, n_latent, z_sample, l_sample, theta, rho, logitpi, qz_logvar, ql_mu, ql_logvar, local_l_mean, local_l_var, eps=1e-6):
	tmp_theta  = theta.view([1,-1])

	elbo_logcll_z       = torch.sum(- 0.5*torch.sum(z_sample**2, axis=1) - n_latent/2 * np.log(2*np.pi))
	tmp_zero            = F.softplus(- logitpi + tmp_theta*torch.log(theta+eps) - tmp_theta*torch.log(tmp_theta+rho+eps)) - F.softplus(-logitpi)
	elbo_logcll_zero    = torch.sum(tmp_zero * (y<eps))
	tmp_nonzero         = -logitpi - F.softplus(-logitpi) \
							+ theta*torch.log(tmp_theta+eps) - theta*torch.log(tmp_theta+rho+eps) \
							+ y*torch.log(rho+eps) - y*torch.log(tmp_theta+rho+eps) \
							+ torch.lgamma(y+tmp_theta+eps) - torch.lgamma(tmp_theta+eps) - torch.lgamma(y+1)
	elbo_logcll_nonzero = torch.sum(tmp_nonzero * (y>eps))

	elbo_entropy_z      = torch.sum(0.5*np.log(2*np.pi) + 0.5*qz_logvar + 0.5)
	kl_l                = torch.sum(0.5*((ql_mu - local_l_mean) ** 2 / local_l_var \
	                                      + torch.exp(ql_logvar) / local_l_var \
	                                      + np.log(local_l_var + eps) \
	                                      - ql_logvar - 1))
	
	logllh              = elbo_logcll_zero + elbo_logcll_nonzero
	elbo                = logllh + elbo_logcll_z + elbo_entropy_z - kl_l

	return logllh, elbo


def fitModel(Y, K, batch_size=128, n_epoch=500, lr=5e-4, non_linear=True, act_nn_exp=True, Y_test=None, eval_train=True):

	t0 = time.time()

	# basic setup
	torch.set_default_dtype(torch.float32)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	N, D = Y.shape
	if (batch_size == N):
		Y_train = torch.from_numpy(np.array(Y)).to(device)
	iterep = int(N / float(batch_size))

	# test setup
	if (not Y_test is None):
		N_test, _ = Y_test.shape
		log_library_size_test = np.log(np.sum(Y_test, axis=1))
		local_l_mean_test, local_l_var_test = np.mean(log_library_size_test), np.var(log_library_size_test)
		Y_test = torch.from_numpy(np.array(Y_test)).to(device)

	# define nets and optimizer
	Enet = Encoder(D, K, non_linear=non_linear).to(device)
	Dnet = Decoder(D, K, act_nn_exp=act_nn_exp).to(device)
	Eoptimizer = optim.Adam(Enet.parameters(), lr=lr)
	Doptimizer = optim.Adam(Dnet.parameters(), lr=lr)

	losses  = []
	epoches = []
	logllhs = []
	times   = []
	losses_test = []

	# train
	for t in range(iterep * n_epoch):

		if (batch_size < N):
			index_train = np.random.choice(np.arange(N), size=batch_size)
			Y_train_np  = Y[index_train, :]
			log_library_size = np.log(np.sum(Y_train_np, axis=1))
			local_l_mean, local_l_var = np.mean(log_library_size), np.var(log_library_size)

			Y_train = torch.from_numpy(Y_train_np).to(device)

		# encoder and decoder
		Enet.train()
		Dnet.train()
		Enet.zero_grad()
		Dnet.zero_grad()
		qz_mu, qz_logvar, z_sample, ql_mu, ql_logvar, l_sample = Enet(Y_train)
		theta, rho, logitpi = Dnet(z_sample, l_sample)
		logllh_v, elbo_v = elbo(Y_train, D, K, z_sample, l_sample, theta, rho, logitpi, qz_logvar, ql_mu, ql_logvar, local_l_mean, local_l_var)

		loss = - elbo_v / batch_size
		logllh_v = logllh_v / batch_size
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
					qz_mu, qz_logvar, z_sample, ql_mu, ql_logvar, l_sample = Enet(Y_test)
					theta, rho, logitpi = Dnet(z_sample, l_sample)
					logllh_v, elbo_v = elbo(Y_test, D, K, z_sample, l_sample, theta, rho, logitpi, qz_logvar, ql_mu, ql_logvar, local_l_mean_test, local_l_var_test)
					loss_test = - elbo_v / N_test
					losses_test.append(loss_test.item())

	# evaluate training
	result = {}
	if eval_train:
		with torch.no_grad():
			log_library_size = np.log(np.sum(Y, axis=1))
			local_l_mean, local_l_var = np.mean(log_library_size), np.var(log_library_size)
			Y_train = torch.from_numpy(np.array(Y)).to(device)
			Enet.eval()
			Dnet.eval()
			qz_mu, qz_logvar, z_sample, ql_mu, ql_logvar, l_sample = Enet(Y_train)
			theta, rho, logitpi = Dnet(z_sample, l_sample)
			logllh_v, elbo_v = elbo(Y_train, D, K, z_sample, l_sample, theta, rho, logitpi, qz_logvar, ql_mu, ql_logvar, local_l_mean, local_l_var)
			loss = - elbo_v / N
			logllh_v = logllh_v / N

			result["loss"] = loss.cpu().numpy()
			result["logllh"] = logllh_v.cpu().numpy()
			result["latent"] = z_sample.cpu().numpy()
			result["theta"] = theta.cpu().numpy()
			result["rho"] = rho.cpu().numpy()
			result["logitpi"] = logitpi.cpu().numpy()
			result["W1"] = Dnet.W1_netp.cpu().numpy()
			result["b1"] = Dnet.b1_netp.cpu().numpy()
			result["W2"] = Dnet.W2_netp.cpu().numpy()
			result["b2"] = Dnet.b2_netp.cpu().numpy()
	result["losses"] = losses
	result["epochs"] = epoches
	result["logllhs"] = logllhs
	result["times"] = times

	# evaluate testing
	result_test = {}
	if (not Y_test is None):
		with torch.no_grad():
			Enet.eval()
			Dnet.eval()
			qz_mu, qz_logvar, z_sample, ql_mu, ql_logvar, l_sample = Enet(Y_test)
			theta, rho, logitpi = Dnet(z_sample, l_sample)
			logllh_v, elbo_v = elbo(Y_test, D, K, z_sample, l_sample, theta, rho, logitpi, qz_logvar, ql_mu, ql_logvar, local_l_mean_test, local_l_var_test)
			logllh_v = logllh_v / N_test
			loss = - elbo_v / N_test
			result_test["logllh"] = logllh_v.item()
			result_test["loss"] = loss.item()
			result_test["latent"] = z_sample.cpu().numpy()
	result_test["losses"] = losses_test 

	return result, result_test


