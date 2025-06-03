import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)

cdef str _num_transfer(str seq, str seqtype):
    if seqtype == "DNA":
        seq = seq.replace("A", "0").replace("C", "1").replace("G", "2").replace("T", "3")
    elif seqtype == "ATCGN":
        seq = seq.replace("A", "0").replace("C", "1").replace("G", "2").replace("T", "3").replace("N", "4")
    elif seqtype == "Mutation":
        seq = seq.replace("A", "0").replace("C", "1").replace("G", "2").replace("T", "3").replace("H", "4").replace("I", "5").replace("J", "6").replace("K", "7")
        seq = seq.replace("L", "8").replace("M", "9").replace("N", "A").replace("O", "B").replace("P", "C").replace("Q", "D").replace("R", "E").replace("S", "F")
    elif seqtype == "RNA":
        seq = seq.replace("A", "0").replace("C", "1").replace("G", "2").replace("U", "3")
    seq = ''.join(filter(lambda x: x in "0123456789ABCDEF", seq.upper()))

    return seq    


cdef list _num_transfer_loc(str num_seq, int K, str seqtype):
    cdef list loc
    cdef int base
    loc = []
    if seqtype == "Mutation":
        base = 16  # 使用 16 进制
    elif seqtype == "ATCGN":
        base=5
    else:
        base = 4   # 使用 4 进制
    for i in range(0, len(num_seq)-K+1):
        #print(f"num_seq[i:i+K] = {num_seq[i:i+K]}")
        loc.append(int(num_seq[i:i+K], base))
    
    return loc

cdef np.ndarray[np.float64_t, ndim=1] _loc_transfer_feature(list loc_list, int K, str seqtype):
    cdef np.ndarray[np.float64_t, ndim=1] feature
    cdef int matrix_size
    cdef double num
    if seqtype == "Mutation":
        matrix_size = int(16**K)
    elif seqtype == "ATCGN":
        matrix_size = int(5**K)
    else:
        matrix_size = int(4**K)
    feature = np.zeros(matrix_size, dtype=np.float64)
    num = 0
    for loc in loc_list:
        feature[loc] += 1
        num+=1

    feature = feature / num

    return feature * 100

cdef np.ndarray[np.float64_t, ndim=1] _loc_transfer_matrix(list loc_list, list dis_list, int K, int length, str seqtype):
    cdef np.ndarray[np.float64_t, ndim=2] matrix
    cdef np.ndarray[np.float64_t, ndim=1] new_matrix
    cdef double num
    cdef int matrix_size
    if seqtype == "Mutation":
        matrix_size = int(16**K)
    elif seqtype == "ATCGN":
        matrix_size = int(5**K)
    else:
        matrix_size = int(4**K)
    #print(f"matrix_size = {matrix_size}, type(matrix_size) = {type(matrix_size)}")
    matrix = np.zeros((matrix_size, matrix_size), dtype=np.float64)
    num = 0
    for dis in dis_list:
        for i in range(0, len(loc_list)-K-dis):
            #print(f"loc_list[i] = {loc_list[i]}")
            #print(f"loc_list[i+K+dis] = {loc_list[i+K+dis]}")
            matrix[loc_list[i]][loc_list[i+K+dis]] += 1
        num = num + (length - 2*K - dis + 1.0)

    #print(f"matrix = {matrix}")

    matrix = matrix / num
    
    #new_matrix = matrix.flatten()
    
    #print(f"new_matrix = {new_matrix}")

    #return new_matrix
    return matrix

cdef np.ndarray[np.float64_t, ndim=1] _matrix_encoding(str seq, int K, int d, str seqtype):
    cdef int length
    cdef np.ndarray[np.float64_t, ndim=1] feature,node_feature
    cdef str num_seq
    cdef list loc, dis
    seq = seq.upper()
    length = len(seq)
    
    num_seq = _num_transfer(seq, seqtype)
    loc = _num_transfer_loc(num_seq, K, seqtype)
    #print(f"loc = {loc}")
    node_feature = _loc_transfer_feature(loc, K, seqtype)
    #print(f"node_feature = {node_feature}")

    dis = [list(range(0, 1)), list(range(1, 2)), list(range(2, 3)),
            list(range(3, 5)), list(range(5, 9)), list(range(9, 17)), list(range(17, 33)),
            list(range(33, 65))]
    if d == 1:
        feature = np.hstack((_loc_transfer_matrix(loc, list(range(0, 1)), K, length, seqtype)))
    
    elif d == 2:
        feature = np.hstack((
            _loc_transfer_matrix(loc, list(range(0, 1)), K, length, seqtype),
            _loc_transfer_matrix(loc, list(range(1, 2)), K, length, seqtype)))
    else:
        feature = np.hstack((
            _loc_transfer_matrix(loc, list(range(0, 1)), K, length, seqtype),
            _loc_transfer_matrix(loc, list(range(1, 2)), K, length, seqtype)))
        for i in range(2, d):
            feature = np.hstack((feature, _loc_transfer_matrix(loc, dis[i], K, length, seqtype)))
     
    feature = np.hstack((feature,node_feature))
    return feature * 100

cpdef tuple create_graph_data(str seq, int K, int d, str seqtype):
    cdef str num_seq
    cdef list loc, dis
    cdef np.ndarray[np.float64_t, ndim=1] node_feature
    cdef np.ndarray[np.float64_t, ndim=2] kmer_features
    cdef np.ndarray[np.float64_t, ndim=2] x
    cdef np.ndarray[np.int64_t, ndim=2] edge_index
    cdef np.ndarray[np.float64_t, ndim=2] edge_attr
    cdef np.ndarray[np.float64_t, ndim=2] matrix
    cdef int i, j,ds
    cdef double matrix_size

    if seqtype == "ATCGN":
        matrix_size = 5**K
    elif seqtype == "Mutation":
        matrix_size = 16**K
    else:
        matrix_size = 4**K

    seq = seq.upper()
    #print(f"seq = {seq}")
    length = len(seq)
    num_seq = _num_transfer(seq, seqtype)
    #print(f"num_seq = {num_seq}")
    loc = _num_transfer_loc(num_seq, K, seqtype)
    node_feature = _loc_transfer_feature(loc, K, seqtype)

    # 生成所有可能的 k-mer 组合的 One-Hot 编码
    kmer_features = generate_all_kmer_features(K, seqtype)
    #print(f"kmer_features = {kmer_features}")

    x = np.hstack([node_feature.reshape(-1, 1), kmer_features])
    #print(f"x = {x}")

    if seqtype == "ATCGN":
        matrix_size = 5**K
    elif seqtype == "Mutation":
        matrix_size = 16**K
    else:
        matrix_size = 4**K

    # 初始化三维特征张量 [i,j,ds]
    edge_attr_3d = np.zeros((int(matrix_size), int(matrix_size), d), dtype=np.float64)

    # 填充各距离层特征
    dis = [list(range(0,1)), list(range(1,2)), list(range(2,3)),
           list(range(3,5)), list(range(5,9)), list(range(9,17)),
           list(range(17,33)), list(range(33,65))]
    
    for ds in range(d):
        current_dis = dis[ds] if ds < len(dis) else []
        matrix = _loc_transfer_matrix(loc, current_dis, K, length, seqtype)
        edge_attr_3d[:, :, ds] = matrix

    # 构建边索引和属性
    edge_list = []
    attr_list = []
    for i in range(int(matrix_size)):
        for j in range(int(matrix_size)):
            if np.any(edge_attr_3d[i,j] > 0):
                edge_list.append([i, j])
                attr_list.append(edge_attr_3d[i,j])

    # 转换为numpy数组
    edge_index = np.array(edge_list, dtype=np.int64).T
    edge_attr = np.array(attr_list, dtype=np.float64)
    edge_attr = edge_attr * 100
    return x, edge_index, edge_attr

def generate_all_kmer_features(int K, str seqtype):
    cdef dict mapping
    cdef int base
    cdef list char_map
    cdef int num_kmers
    cdef np.ndarray[np.float64_t, ndim=2] kmer_features
    cdef int i, remainder, pos

    # 初始化映射关系和字符表
    if seqtype == "DNA":
        base = 4
        char_map = ['A', 'C', 'G', 'T']
    elif seqtype == "ATCGN":
        base = 5
        char_map = ['N', 'A', 'C', 'G', 'T']
    elif seqtype == "Mutation":
        base = 16
        char_map = [str(i) for i in range(16)]
    elif seqtype == "RNA":
        base = 4
        char_map = ['A', 'C', 'G', 'U']

    num_kmers = base ** K
    # 特征维度调整为 base*K（平铺后的形状）
    kmer_features = np.zeros((num_kmers, base * K), dtype=np.float64)

    for i in range(num_kmers):
        kmer = ""
        val = i
        # 生成k-mer字符串
        for _ in range(K):
            remainder = val % base
            kmer = char_map[remainder] + kmer
            val = val // base
        
        # 生成平铺的独热编码
        encoded = []
        for char in kmer:
            # 找到字符在映射表中的位置
            pos = char_map.index(char)
            # 生成独热编码并平铺
            one_hot = [0.0] * base
            one_hot[pos] = 1.0
            encoded.extend(one_hot)
        
        kmer_features[i] = encoded

    return kmer_features

def matrix_encoding(seq, K, d, seqtype):

    return _matrix_encoding(seq, K, d, seqtype)

