def compute_padding(M, N):
    M_padded = ((M + 2 ** (self.J)) // 2 ** self.J + 1) * 2 ** self.J
    N_padded = ((N + 2 ** (self.J)) // 2 ** self.J + 1) * 2 ** self.J

    return M_padded, N_padded
