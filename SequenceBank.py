

###############################################################
#
#   SequenceBank
#

class SequenceBank:

    def __init__(self, u, q_size):

        # q_size support 4, 3, 2
        assert (q_size == 4) or (q_size == 3) or (q_size == 2)

        self._q_size = q_size

        l_data = []

        for x in range (u._next_free):
            data = u.get_data_at_loc(x)

            if data.shape[0] >= self._q_size:
                aT = rolling_window(data, self._q_size)
                l_data.append(aT)
      
        data = np.vstack(l_data)

        self.uniques_map_to_dense = np.unique(data)

        sorted = np.searchsorted(self.uniques_map_to_dense, data)

        assert (self.uniques_map_to_dense[sorted] == data).all()

        self._factor = sorted.max() + 1

        assert np.log2(self._factor)* 4 < 64

        if self._q_size == 4:
            a = sorted[:, 0] + self._factor * (sorted[:, 1] + self._factor * (sorted[:, 2] + self._factor * (sorted[:,3])))

        if self._q_size == 3:
            a = sorted[:, 0] + self._factor * (sorted[:, 1] + self._factor * (sorted[:, 2]))

        if self._q_size == 2:
            a = sorted[:, 0] + self._factor * sorted[:, 1]
        

        self._unique, self._count = np.unique(a, return_counts = True)

    def predict(self, l_qa, q):

        a_test = np.array(l_qa)

        print(a_test, q)

        l_y = []
        for a in range (5):

            qa_final = q * 5 + a

            dense_qa_final = np.searchsorted(self.uniques_map_to_dense, qa_final)

            if self.uniques_map_to_dense[dense_qa_final] != qa_final:
                print(a, f"  Not found. Final QA q = {q} a = {a}, qa = {qa_final} not existing in dataset")
                l_y.append(0)
                continue

            dense_qa = np.searchsorted(self.uniques_map_to_dense, l_qa)

            if self.uniques_map_to_dense[dense_qa] != l_qa:
                print(a, "  Not found. QA sequence not existing in dataset")
                l_y.append(0)
                continue


            if self._q_size == 4:
                a_packed = dense_qa[0] + self._factor * (dense_qa[1] + self._factor * (dense_qa[2] + self._factor * (dense_qa_final)))

            if self._q_size == 3:
                 a_packed = dense_qa[0] + self._factor * (dense_qa[1] + self._factor * (dense_qa_final))

            if self._q_size == 2:
                 a_packed = dense_qa[0] + self._factor * dense_qa_final


            idx = np.searchsorted(self._unique, a_packed)

            if (idx < self._unique.shape[0]) and (self._unique[idx] == a_packed):
                l_y.append(self._count[idx])
                print (a, self._unique[idx], self._count[idx])
            else:
                print(a, "  Not found")
                l_y.append(0)

        return l_y

