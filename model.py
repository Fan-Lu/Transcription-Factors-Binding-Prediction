import numpy as np

class Functions:
    def OneHotEncoding(self, lines, len_kernel, no_samples):
        A = np.array([[1.0, 0.0, 0.0, 0.0]])
        C = np.array([[0.0, 1.0, 0.0, 0.0]])
        G = np.array([[0.0, 0.0, 1.0, 0.0]])
        T = np.array([[0.0, 0.0, 0.0, 1.0]])

        All_S_Test = 0.25 * np.ones((no_samples, 400 + len_kernel * 8), dtype=np.float32)
        All_S_Test[:, len_kernel*4:len_kernel*4+400] = np.zeros((no_samples, 400), dtype=np.float32)

        for k in range(0, no_samples):
            in_seq = lines[K]

            for j in range(0, 100):
                if in_seq[j] == 'A':
                    All_S_Test[k, (len_kernel + j) * 4:(len_kernel + j + 1) * 4] = A
                if in_seq[j] == 'C':
                    All_S_Test[k, (len_kernel + j) * 4:(len_kernel + j + 1) * 4] = C
                if in_seq[j] == 'G':
                    All_S_Test[k, (len_kernel + j) * 4:(len_kernel + j + 1) * 4] = G
                if in_seq[j] == 'T':
                    All_S_Test[k, (len_kernel + j) * 4:(len_kernel + j + 1) * 4] = T

        return All_S_Test

    def PerformanceEval(self, y_out, test_label, no_samples):
        eval_metric = np.zeros((no_samples, 4), dtype=np.float)

        return eval_metric