

#################################################################
#
#       UserBank
#

class UserBank:

    def store_fast(self, anID, anValue):

        iSort = np.argsort(anID, kind = 'stable')

        anID = anID[iSort]
        anValue = anValue[iSort]

        anTest, idx_start, count = np.unique(anID, return_counts= True, return_index= True)

        self.ensure_users(anTest)
        
        count_max = np.max(count)

        for iCount in range (0, count_max):

            m = (count >= (iCount + 1))
       
            idx_value = idx_start[m] + iCount

            anID_current = anID[idx_value]
            anValue_current = anValue[idx_value]

            aiLoc = self.get_users_must_exist(anID_current)

            iOffset = aiLoc * self._width + self._anNext[aiLoc]

            self._anData.ravel()[iOffset] = anValue_current

            self._anNext[aiLoc] = (self._anNext[aiLoc] + 1) % self._width

    def get_users_must_exist(self, test_user):
        aIdx = np.searchsorted(self._anUser[:self._next_free], test_user)
        return self._anLoc[aIdx]

    def get_users(self, test_user):
        idx = np.searchsorted(self._anUser[:self._next_free], test_user)
        m_user_range = (idx < self._capacity)

        # Place anywhere in range. These are invalidated.
        idx[~m_user_range] = 0

        m_userfound = (self._anUser[idx] == test_user)

        m_userfound = m_userfound & m_user_range

        idx[~m_userfound] = 0

        anLoc = (self._anLoc[idx]).astype(np.int32)

        anLoc[~m_userfound] = -1

        return anLoc


    def ensure_users(self, test_user):

        idx = np.searchsorted(self._anUser, test_user)
        
        m_user_range = (idx < self._capacity)

        idx = idx[m_user_range]

        m_userfound = (self._anUser[idx] == test_user[m_user_range])

        new_users = np.concatenate([test_user[~m_user_range], test_user[m_user_range][~m_userfound]])
        num_new_users = new_users.shape[0]

        assert self._next_free <= self._capacity, "UserBank capacity logic error"

        num_free = self._capacity - self._next_free

        if num_free < num_new_users:
            needAdditional = num_new_users - num_free
            self.resize(needAdditional)


        num_free = self._capacity - self._next_free
        assert num_free >= num_new_users

        self._anUser[self._next_free: self._next_free + num_new_users] = new_users

        self._anLoc[self._next_free: self._next_free + num_new_users] =  np.arange(self._next_free, self._next_free + num_new_users)

        self._next_free = self._next_free + num_new_users

        iUserSort = np.argsort(self._anUser[:self._next_free])

        self._anUser[:self._next_free] = self._anUser[iUserSort][:self._next_free]
        self._anLoc[:self._next_free] = self._anLoc[iUserSort][:self._next_free]
        

    def init_memory(self):
        self._anUser = np.empty(self._capacity, dtype = np.uint32)
        self._anUser[:] = np.iinfo(np.uint32).max

        self._anLoc = np.empty(self._capacity, dtype = np.uint32)
        self._anLoc[:] = np.iinfo(np.uint32).max

        self._anNext = np.zeros((self._capacity), dtype=np.uint16)
        self._anData = np.zeros((self._capacity, self._width), dtype=self._datatype)

        self._anData[:, :] = np.iinfo(self._datatype).max

    def __init__(self, df, width, datatype):
        self._width = width

        self._datatype = datatype

        if df is None:
            self._capacity = 650000
            self.init_memory()
            self._next_free = 0
        else:
            anUser = np.array(df.user_id)
            anILoc = np.arange(anUser.shape[0])

            x = df.sData.apply(lambda x: np.array(x[-self._width:]))

            l = x.map(lambda x: x.shape[0])

            iNext = l % self._width

            x = x.map(lambda x: np.pad(x, mode = 'constant', constant_values = np.iinfo(self._datatype).max, pad_width = (0, self._width - x.shape[0])))

            an = np.array(x.values.tolist(), dtype = self._datatype)

            self._anUser = anUser
            self._anLoc = anILoc

            self._anNext = iNext.astype(np.uint16)
            self._anData = an

            self._next_free = anUser.shape[0]
            self._capacity = self._next_free

            iUserSort = np.argsort(self._anUser[:self._next_free])

            self._anUser[:self._next_free] = self._anUser[iUserSort][:self._next_free]
            self._anLoc[:self._next_free] = self._anLoc[iUserSort][:self._next_free]


    def resize(self, ensureNumNew = 0):
        old_capacity = self._capacity
        self._capacity = int(1.2 * old_capacity) + ensureNumNew

        print(f"UserBank resizing from {old_capacity} to {self._capacity}")

        anUser = self._anUser
        anLoc = self._anLoc
        anNext = self._anNext
        anData = self._anData

        self.init_memory()

        self._anUser[:self._next_free] = anUser[:self._next_free]
        self._anLoc[:self._next_free] = anLoc[:self._next_free]
        self._anNext[:self._next_free] = anNext[:self._next_free]
        self._anData[:self._next_free] = anData[:self._next_free]

        gc.collect()

    def get_user(self, test_user):
        idx = np.searchsorted(self._anUser[:self._next_free], test_user)
        userFound =  (idx < self._capacity) and (self._anUser[idx] == test_user)
        
        if userFound:
            return self._anLoc[idx]
        else: 
            return -1


    def get_user_create_if_not_exists(self, test_user):
        idx = np.searchsorted(self._anUser[:self._next_free], test_user)
        userFound =  (idx < self._capacity) and (self._anUser[idx] == test_user)

        if userFound:
            return self._anLoc[idx]
        else:                    
            assert self._next_free <= self._capacity, "UserBank capacity logic error"
            if self._next_free == self._capacity:
                self.resize()
            
            self._anUser[self._next_free] = test_user
            self._anLoc[self._next_free] = self._next_free
            self._next_free = self._next_free + 1

            iUserSort = np.argsort(self._anUser[:self._next_free])

            self._anUser[:self._next_free] = self._anUser[iUserSort][:self._next_free]
            self._anLoc[:self._next_free] = self._anLoc[iUserSort][:self._next_free]

            return self._next_free - 1

    def add(self, d):

        for (user_id, l_content) in d.items():
            iLoc = self.get_user_create_if_not_exists(user_id)

            iNext = self._anNext[iLoc]

            for value in l_content:
                self._anData[iLoc][iNext] = value
                iNext = (iNext + 1) % (self._width)

            self._anNext[iLoc] = iNext

    def get_data_at_loc(self, iLoc):

        data = self._anData[iLoc]
        iNext = self._anNext[iLoc]

        data = np.concatenate([data[iNext :], data[:iNext]])

        data = data[data < np.iinfo(self._datatype).max]

        return data


    def get_data_for_user(self, user_id):
        iLoc = self.get_user(user_id)
        if iLoc >= 0:
            return self.get_data_at_loc(iLoc)
        else:
            return None


    def viz_num_data(self):
        
        num_users = u._next_free

        l_data_len = []

        for x in range(num_users):
            l_data_len.append(self.get_data_at_loc(x).shape[0])

        anLen = np.array(l_data_len)
        sns.distplot(anLen, bins = self._width)
        fla


#################################################################
#
#       userbanks_equal
#

def userbanks_equal(u0, u1):

    anUser0 = u0._anUser[:u0._next_free]
    anUser1 = u1._anUser[:u1._next_free]

    if u0._next_free != u1._next_free:
        print("Not same length")
        return False

    if (np.unique(anUser0) == np.unique(anUser1)).all():
        pass
    else:
        print("Same length different users")
        return False

    for x in anUser0: 
        user_data_0 = u0.get_data_at_loc(u0.get_user(x))
        user_data_1 = u1.get_data_at_loc(u1.get_user(x))

        if user_data_0.shape != user_data_1.shape:
            print(f"Datasize not equal data for {x}")
            return False

        m_equal = (user_data_0 == user_data_1)

        if not m_equal.all():
            print(f"Not equal data for {x}")
            return False

    return True


#################################################################
#
#       compact_qa
#

def compact_qa(df):
    
    m = df.content_type_id == 0

    df = df[m].reset_index(drop = True)

    sID = df.user_id


    sContentID = df.content_id
    sAnswer = df.user_answer

    sData = (sContentID.astype(np.uint32) * 5) + sAnswer

    # See if should be custom datatype
    assert sData.max() < np.iinfo(np.uint32).max, "Overflow"

    sData = sData.astype(np.uint32)

    df = df.assign(sData = sData)

    df = df.groupby('user_id').sData.apply(list).reset_index()

    return df


