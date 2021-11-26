# cat_cluster.py 
# cluster non-numeric items using category utility
# Anaconda 4.1.1 (Python 3.5)

import numpy as np

def cat_utility(ds, clustering, m):
  # category utility of clustering of dataset ds
  n = len(ds)  # number items
  d = len(ds[0])  # number attributes/dimensions

  # get number items in each cluster
  cluster_cts = [0] * m  # [0,0]
  for ni in range(n):  # each item
    k = clustering[ni]
    cluster_cts[k] += 1

  for i in range(m): 
    if cluster_cts[i] == 0:   # a cluster has no items
      return 0.0

  # get number unique values, each att
  # ex: [3, 3, 2] -> 3 colors, 3 lengths, 2 weights
  # same as max+1 in ds if ds is encoded
  # used only for list allocation
  unique_vals = [0] * d  # [0,0,0]
  for i in range(d):  # each att/dim
    maxi = 0
    for ni in range(n):  # each item
      if ds[ni][i] > maxi: maxi = ds[ni][i]
    unique_vals[i] = maxi+1

  # get number of each value in each att
  # ex: [[2,1,2], [1,3,1], [2,3]] -- 2 red, 1 blue, etc.
  att_cts = []
  for i in range(d): # each att
    cts = [0] * unique_vals[i] 
    for ni in range(n):  # each data item
      v = ds[ni][i]
      cts[v] += 1
    att_cts.append(cts)

  # get number of each value in each att, each cluster
  # ex: k_cts = [ k=0 [[2,0,0], [1,0,1], [1,1]],  
  #               k=1 [[0,1,2], [0,3,0], [1,2]] ]
  k_cts = []
  for k in range(m):  # each cluster
    a_cts = []
    for i in range(d): # each att
      cts = [0] * unique_vals[i] 
      for ni in range(n):  # each data item
        if clustering[ni] != k: continue  # wrong cluster
        v = ds[ni][i]
        cts[v] += 1
      a_cts.append(cts)
    k_cts.append(a_cts) 

  # uncoditional sum squared probs (right summation)
  un_sum_sq = 0.0 
  for i in range(d):  
    for j in range(len(att_cts[i])):
      un_sum_sq += (1.0 * att_cts[i][j] / n) \
      * (1.0 * att_cts[i][j] / n) 

  # conditional sum, each cluster (left summation)
  cond_sum_sq = [0.0] * m  
  for k in range(m):  # each cluster
    sum = 0.0
    for i in range(d):
      for j in range(len(att_cts[i])):
        if cluster_cts[k] == 0: print("FATAL LOGIC ERROR")
        sum += (1.0 * k_cts[k][i][j] / cluster_cts[k]) \
        * (1.0 * k_cts[k][i][j] / cluster_cts[k])
    cond_sum_sq[k] = sum

  # P(C)
  prob_c = [0.0] * m  # [0.0, 0.0]
  for k in range(m):  # each cluster
    prob_c[k] = (1.0 * cluster_cts[k]) / n
  
  # put it all together
  left = 1.0 / m
  right = 0.0
  for k in range(m):
    right += prob_c[k] * (cond_sum_sq[k] - un_sum_sq)
  cu = left * right
  return cu

def cluster(ds, m):
  # ds is encoded
  # greedy algorithm
  n = len(ds)  # number items to cluster

  # assumes first m items are 'different'
  # because they seed the first m clusters
  working_set = [0] * m
  for k in range(m):
    working_set[k] = list(ds[k]) 
    
  clustering = list(range(m))  # [0,1,2, .. m-1]

  for i in range(m, n):
    item_to_cluster = ds[i]
    working_set.append(item_to_cluster)  # working set changed

    proposed_clusterings = []  # empty list
    for k in range(m):         # proposed new clusterings
      copy_of_clustering = list(clustering) 
      copy_of_clustering.append(k)
      proposed_clusterings.append(copy_of_clustering) 

    proposed_cus = [0.0] * m   # compute CU of each proposed
    for k in range(m):
      proposed_cus[k] = \
        cat_utility(working_set, proposed_clusterings[k], m)

    # which proposed clustering will give best CU? (greedy)
    best_proposed = np.argmax(proposed_cus)  # 0, 1, . . m-1

    # update clustering
    clustering.append(best_proposed)

  return clustering

# =======================================

def main():
  print("\nIniciando clusterizacao atraves do metodo C.U. ")

  raw_data = [['1','false','8','2','137.183.95.242','pt-BR','true','Windows 10','true','America/Sao_Paulo','false','Chrome','95.0.4638','nvidia evga geforce gtx 970','1gd7abb2e497c9f5a00ff94da5dc9e2','0.5660000000000023'],
['2','false','4','4','34.131.211.72','pt-BR','true','Windows 8','true','America/Sao_Paulo','false','Firefox','80.0.4170','amd radeon rx 560','1adl54148eac5h2035mbf91h81e8i','0.06000000000000117'],
['3','false','8','2','201.98.244.196','pt-BR','true','Windows 10','true','America/Sao_Paulo','false','Chrome','95.0.4638','nvidia evga geforce gtx 970','e09c80c42fda55f9d992e59ca6b3307d','0.12300000000000122'],
['4','true','4','2','107.5.15.1','pt-BR','true','Windows 10','true','America/Sao_Paulo','false','Edge','80.0.4170','amd radeon rx 560','1gd7abb2e497c9f5a00ff94da5dc9e2','0.7265000000000015'],
['5','true','2','6','251.203.64.179','pt-BR','false','Windows 8','true','America/Sao_Paulo','false','Firefox','80.0.4170','amd radeon rx 560','1gd7abb2e497c9f5a00ff94da5dc9e2','0.6250000000000023'],
['6','false','4','4','138.84.98.53','pt-BR','true','Windows 8','true','America/Sao_Paulo','false','Chrome','95.0.4638','amd radeon rx 560','1gd7abb2e497c9f5a00ff94da5dc9e2','0.07400000000000118'],
['7','true','8','2','201.110.67.113','pt-BR','true','Windows 10','true','America/Sao_Paulo','false','Chrome','80.0.4170','nvidia evga geforce gtx 970','e1faffe9c3c801f2f8c3fbe7cb032cb2','0.07850000000000101'],
['8','false','8','2','18.48.206.162','pt-BR','true','Windows 10','true','America/Sao_Paulo','false','Chrome','95.0.4638','nvidia evga geforce gtx 970','41fcba09f2bdcdf315ba4119dc7978dd','0.15850000000000108'],
['9','true','2','8','165.153.240.253','pt-BR','false','Windows 8','true','America/Sao_Paulo','false','Firefox','80.0.4170','amd radeon rx 560','1gd7abb2e497c9f5a00ff94da5dc9e2','0.5030000000000022'],
['10','true','4','6','701.861.732.26','pt-BR','true','Windows 10','true','America/Sao_Paulo','false','Chrome','80.0.4171','amd radeon rx 260','k1faffe9c3c801f2f8c3fbe7cb032cb1','0.6250000000000023'],
['11','true','4','2','98.169.9.107','pt-BR','true','Windows 10','true','America/Sao_Paulo','false','Chrome','95.0.4639','amd radeon rx 160','9ca980e2204bbc788171d43b6f8b27ee','0.10050000000000081'],
['12','true','16','6','229.4.84.39','en-US','true','Linux','true','America/California','false','Opera','70.0.4120','intel uhd graphics 630','7e94a7a2f62aa616e91256f315ccc343','0.5550000000000023'],
['13','false','8','6','205.28.233.120','en-US','true','Linux','false','America/California','false','Edge','70.0.4121','nvidia geforce gtx 1060','03886aac90d383bd2c82d0c1fee408db','0.5480000000000023'],
['14','true','4','4','236.53.143.136','en-US','false','Android 11','true','America/Arizona','true','Firefox','80.0.4160','Mali-G71 MP2','bd2c63e2202cf77b0db4d23b14ee8acb','0.5635000000000014'],
['15','true','16','20','176.113.181.34','en-US','true','Linux','true','America/Texas','false','Edge','80.0.4161','nvidia geforce gtx 1060','67c8a040082074f7c1da8315bafe0e34','0.7275000000000016'],
['16','true','16','20','9.118.46.170','en-US','false','Linux','true','America/Utah','false','Firefox','80.0.4162','intel uhd graphics 630','ef60e89acc989917b61bd6a053156cf7','0.6045000000000011'],
['17','false','32','20','67.239.57.245','en-US','false','Windows 10','true','America/Texas','false','Firefox','80.0.4163','radeon rx 580','5565ce34d7a4a3d5c9cf1bc49eb84636','0.5905000000000015'],
['18','false','8','6','215.214.145.58','en-US','true','Android 10','true','America/Arizona','true','Chrome','95.0.4638','Adreno 660','b395ffe4ab378636532c48e8fe7cb820','0.5610000000000014'],
['19','false','4','2','223.1.148.150','en-US','false','Android 11','true','America/Lousiana','true','Chrome','95.0.4635','Mali-G78 MP24','34c4dfddcbc0111a0f59ed600b03b78d','0.7620000000000017'],
['20','true','16','12','8.184.26.0','en-US','true','Windows XP','true','America/Alabama','false','Chrome','95.0.4638','radeon rx 580','cd4d8c98455540fd3000f194ae9e6842','0.6585000000000003'],
['21','true','16','12','29231217148','en-US','true','Windows XP','true','America/Georgia','false','Chrome','95.0.4638','intel uhd graphics 640','9fcea301d1ae372716a51752a1ab64d9','0.7370000000000012'],
['22','true','32','20','27.208.218.29','en-GB','true','Windows XP','true','Europe/Oxford','false','Firefox','80.0.4170','intel uhd graphics 650','7f19eaaf1086c3be115e39d6047c6879','0.5085000000000006'],
['23','false','8','4','137.222.17.31','en-GB','true','Linux','true','Europe/Dublin','false','Edge','80.0.4171','intel uhd graphics 640','b31d697139a52828107d52034cbbe8f4','0.5085000000000006'],
['24','true','16','12','58.46.11.80','en-GB','true','Linux','true','Europe/Dublin','false','Opera?','70.0.4120','intel uhd graphics 650','69194f9fc78b77dc5f550a7c809c17d6','0.5945000000000008'],
['25','true','8','6','131.26.244.97','en-GB','false','Android 11','true','Europe/London','true','Chrome','95.0.4638','Adreno 612','8959bd046ead35c66091b680eed092e6','0.7435000000000017'],
['26','true','8','6','131.26.244.97','en-GB','false','Android 11','true','Europe/London','true','Chrome','95.0.4638','Adreno 612','8959bd046ead35c66091b680eed092e6','0.5805000000000016'],
['27','true','8','6','131.26.244.97','en-GB','false','Android 11','true','Europe/London','true','Chrome','95.0.4638','Adreno 612','8959bd046ead35c66091b680eed092e6','-0.43299999999999983'],
['28','true','8','6','131.26.244.97','en-GB','false','Android 11','true','Europe/London','true','Chrome','95.0.4638','Adreno 612','8959bd046ead35c66091b680eed092e6','-0.43299999999999983'],
['29','true','8','4','91.240.97.57','en-GB','true','Windows 8','true','Europe/Cardiff','false','Firefox','80.0.4168','radeon rx 6600 xt','6d39163fcbe897048370d8b6fa233032','-0.43299999999999983'],
['30','true','8','4','39.235.12.240','en-GB','true','IOS 15?','true','Europe/Cardiff','true','Safari','61.4.5080','A13 Bionics GPU','8bb08e71ffa369ed71e504a749d9a0be','-0.43299999999999983'],
['31','true','8','6','231.164.15.55','pt-PT','false','Windows 7','true','Europe/Evora','false','Edge','61.4.5081','nvidia evga geforce gtx 970','0bad43bfbe903abd6d0ea379ddef8138','0.5875000000000007'],
['32','false','8','6','89.99.27.50','pt-PT','false','Windows 7','true','Europe/Evora','false','Opera?','70.0.4120','intel uhd graphics 630','d7d8de4a4c8d3899a1bcd3db71912dca','0.6030000000000004'],
['33','false','8','4','166.60.80.49','pt-PT','true','Windows 10','true','Europe/Aveiro','false','Firefox','80.0.4167','nvidia evga geforce gtx 970','cb87064afb218423b3f0d8be57ddf118','0.5685000000000002'],
['34','true','16','12','220.128.252.71','pt-PT','false','Android 10','true','Europe/Braga','true','Chrome','95.0.3590','Radeon HD 7610M','a904adf1911e772becd62cf072a37b94','0.7065000000000003'],
['35','false','8','4','166.60.80.49','pt-PT','true','Windows 10','true','Europe/Aveiro','false','Firefox','80.0.4167','nvidia evga geforce gtx 970','cb87064afb218423b3f0d8be57ddf118','0.5615000000000014'],
['36','false','16','8','101.173.153.91','pt-PT','true','IOS 15?','false','Europe/Coimbra','true','Safari','61.3.5209','A12 Bionics GPU','c111269f24d7e7515236abf18f861106','0.5705'],
['37','false','4','2','105.59.86.48','pt-PT','true','Android 11','false','Europe/Coimbra','true','Chrome','61.3.5210','Adreno 660','d0730cfd4d827fef106139286e877bd7','0.5615000000000014'],
['38','true','8','6','131.26.244.97','en-GB','false','Android 11','true','Europe/London','true','Chrome','95.0.4638','Adreno 612','8959bd046ead35c66091b680eed092e6','0.7935000000000003'],
['39','true','8','6','131.26.244.97','en-GB','false','Android 11','true','Europe/London','true','Chrome','95.0.4638','Adreno 612','8959bd046ead35c66091b680eed092e6','0.7340000000000011'],
['40','true','16','12','100.157.3.182','zh-CN','false','Windows 10','true','Asia/Hubei','false','Firefox','95.0.4639','intel uhd graphics 630','8ce781c530d97f1232bd49c44e916d7d','-0.43299999999999983'],
['41','true','32','12','206.204.54.102','zh-CN','false','Linux','true','Asia/Hubei','false','Chrome','95.0.4638','nvidia geforce rtx 3080 ti','7a116881439b348c35038e90952066bd','-0.43299999999999983'],
['42','true','16','20','27.37.85.98','zh-CN','false','Windows 10','true','Asia/Tibet','false','Opera?','70.0.4121','nvidia geforce rtx 2060','d4120332d8711559ac337a5ced66a0b6','0.3355000000000012'],
['43','false','32','20','192.219.178.42','zh-CN','false','Windows 10','true','Asia/Tibet','false','Firefox','70.0.4122','nvidia geforce gtx 1060','f1fd08c30935aeb332f7d422219d69d0','0.39600000000000113'],
['44','false','32','12','127.192.211.72','zh-CN','false','Linux','true','Asia/Tibet','false','Chrome','95.0.4638','nvidia geforce rtx 3080 ti','7a116881439b348c35038e90952066bd','0.5900000000000015'],
['45','true','32','16','2,44E+11','zh-CN','true','Android 11','true','Asia/Guizhu','true','Edge','95.0.4639','Mali-G52 MC2','d79cf52bdd85f83d30bc7ad44df066d2','0.5500000000000017'],
['46','false','32','20','136.211.214.7','zh-CN','true','Linux','false','Asia/Yunnan','false','Chrome','95.0.4638','nvidia geforce rtx 3080 ti','99d93ec623174904b430e24ba8b541e8','0.5555000000000015'],
['47','true','32','20','33.66.240.186','zh-CN','true','Linux','true','Asia/Qinghai','false','Firefox','95.0.4639','nvidia geforce rtx 3080 ti','99d93ec623174904b430e24ba8b541e8','0.608500000000001'],
['48','true','16','12','19.24.171.117','zh-CN','true','Android 11','true','Asia/Qinghai','true','Chrome','95.0.4640','Mali-T830 MP1','6a14b786d4da0994b2eb4c89afb5c687','0.5635000000000014'],
['49','false','16','12','184.61.21.20','zh-TW','false','Windows XP','true','Asia/New_Taipei','false','Firefox','95.0.4641','nvidia geforce rtx 2060','42323e3211ed4478b2b8ba87d4185a03','0.4690000000000013'],
['50','false','32','20','2,05E+11','zh-TW','false','Android 8','true','Asia/New_Taipei','true','Chrome','95.0.4642','Mali-450 MP4','7b60e3847f7a2c3c9621b9ff06587460','0.553000000000001'],
['51','false','32','20','244.57.159.2','zh-TW','false','Windows XP','true','Asia/Taipei','false','Firefox','95.0.4643','nvidia geforce gtx 1060','f1fd08c30935aeb332f7d422219d69d0','0.744500000000001'],
['52','true','16','12','211.107.79.212','zh-TW','false','Linux','true','Asia/Taipei','false','Chrome','95.0.4638','intel uhd graphics 630','76abc7de1cf3cd270e46feb95055fa13','0.7415000000000006'],
['53','true','32','20','123.37.196.184','zh-TW','false','Linux','true','Asia/Taipei','false','Firefox','95.0.4639','nvidia geforce gtx 1060','f1fd08c30935aeb332f7d422219d69d0','0.5545000000000012'],
['54','true','16','6','80247137241','zh-TW','false','Linux','true','Asia/Taipei','false','Opera?','70.0.4118','GeForce GTX 770','3c7b4b210ac26693717cf8789ac40e1b','0.4670000000000015'],
['55','true','32','20','212.237.85.254','zh-TW','true','Android 9','true','Asia/Yilan','true','Opera?','70.0.4110','Mali-T830 MP1','f96f688f2345e3d442e75fb611212c0c','0.37000000000000144'],
['56','true','8','6','216.22.162.184','zh-TW','false','Linux','true','Asia/Yilan','false','Chrome','95.0.4638','intel uhd graphics 630','b19d4f601bf18955a32f47fababcafc1','0.5840000000000017'],
['57','true','16','12','240.23.151.249','zh-TW','false','Windows 10','true','Asia/Yilan','false','Opera?','70.0.4001','intel uhd graphics 630','3d586b693b539a353472277025e055ac','0.6075000000000005'],
['58','false','8','6','126.26.22.172','zh-TW','false','Linux','false','Asia/Yilan','false','Opera?','70.0.4001','nvidia geforce rtx 2080 super','ef166bf43fbf24ef1cb8a78f35c6ffff','0.4530000000000015'],
['59','true','8','6','193.123.6.158','zh-TW','false','Windows 10','true','Asia/Taichung','false','Opera?','70.0.4001','nvidia geforce rtx 2080 super','c8c95be63639567a37746f19334881f3','0.5405000000000018'],
['60','false','16','12','202.66.180.234','zh-TW','true','Android 11','true','Asia/Taichung','true','Edge','70.0.4002','Mali-T830 MP1','533b4b26c116ddcf14b958bf85c49049','0.7110000000000019'],
['61','true','16','8','49202108117','zh-TW','true','IOS 15','true','Asia/Taichung','true','Safari','60.3.5110','A11 Bionics GPU','179be100ef18a605d8bc75393cdac029','0.5510000000000017'],
['62','true','32','20','28.169.29.206','ja','false','Linux','true','Asia/Osaka','false','Opera?','70.0.4001','nvidia geforce rtx 2080 super','9acebce5e4cefc2d4a147defba33d778','0.7485000000000015'],
['63','true','32','12','75219228205','ja','false','Linux','true','Asia/Osaka','false','Chrome','95.0.4638','GeForce4 440','44fda666839f1eb3d761b1f773052ade','0.6310000000000004'],
['64','true','32','20','246.143.255.67','ja','false','Linux','true','Asia/Kobe','false','Opera?','70.0.4001','nvidia geforce rtx 2080 super','9acebce5e4cefc2d4a147defba33d778','0.4850000000000014'],
['65','true','8','4','241.38.210.152','ja','false','Windows 10','true','Asia/Kobe','false','Edge','70.0.4002','nvidia geforce rtx 3080 ti','7a7fd58f09acc749d74f2b1063672bf2','0.5080000000000013'],
['66','false','8','4','1,22E+11','ja','false','Windows 10','true','Asia/Kobe','false','Chrome','95.0.4638','nvidia geforce rtx 3080 ti','7a7fd58f09acc749d74f2b1063672bf2','0.48450000000000143'],
['67','false','16','12','49.0.217.117','ja','false','Windows 10','true','Asia/Kyoto','false','Chrome','95.0.4638','intel uhd graphics 630','8ce781c530d97f1232bd49c44e916d7d','0.4660000000000014'],
['68','true','16','6','156.190.81.174','ja','false','Windows 10','true','Asia/Kyoto','false','Firefox','95.0.4639','radeon rx 550 2gb','51de908885e926c3f433b9a93eaa1e23','0.5360000000000016'],
['69','true','32','12','83.79.255.253','ja','false','Linux','true','Asia/Tokyo','false','Chrome','95.0.4640','nvidia geforce rtx 3080 founders','3313fa3260d2bb02b70be057a8b32fc0','0.4295000000000015'],
['70','true','32','12','3.27.151.153','ja','false','Linux','true','Asia/Tokyo','false','Firefox','95.0.4641','nvidia geforce rtx 3080 founders','3313fa3260d2bb02b70be057a8b32fc0','0.5565000000000014'],
['71','true','8','4','40241158241','fr','false','Windows XP','true','Europe/Marselle','false','Chrome','95.0.4638','Radeon 535','e1e09726f0079dfd12d0f652531a6457','0.4480000000000013'],
['72','false','8','4','54.94.202.59','fr','true','Android 11','false','Europe/Bordeaux','true','Chrome','95.0.4639','Mali-T830 MP1','0439f23e45aacfc84f44a9d119b646ff','0.48200000000000137'],
['73','true','8','6','127.183.46.194','fr','true','IOS 15?','true','Europe/Lyon','true','Safari','60.3.5000','PowerVR 7XT GT7600 Plus','8cb8c802f5012096b8ce0552a7be4a6e','0.5040000000000006'],
['74','true','16','12','25.6.2.161','fr','true','Windows 7','true','Europe/Toulouse','false','Opera?','70.0.4001','Radeon HD 4850','38135692e057265db52833c0b4d1df32','0.687000000000001'],
['75','true','16','12','1,34E+11','fr','true','Android 8','true','Europe/Nice','true','Chrome','70.0.4002','Mali-G52 MC2','04c581e8d109083f8e7be0423a17b223','0.6005000000000001'],
['76','true','16','6','218.58.12.139','fr','true','Android 8','false','Europe/Nantes','true','Chrome','70.0.4003','PowerVR GE8320','9c9f49d6a6343c186e586df87436811f','0.5905000000000002'],
['77','true','8','4','84.24.154.17','fr','true','Windows 8','false','Europe/Nantes','false','Chrome','95.0.4638','Intel 82915G Express','4d723b71df381bce95d41cf72d55c52c','0.5575000000000001'],
['78','false','8','4','182.254.28.113','fr','true','Windows 8','false','Europe/Marselle','false','Chrome','95.0.4638','Intel 82915G Express','4d723b71df381bce95d41cf72d55c52c','0.5580000000000002'],
['79','true','16','12','18.234.41.224','cs','false','Linux','true','Europe/Vysocina','false','Chrome','95.0.4532','intel uhd graphics 630','a684546f9f737ab373710808c7a438f4','0.39750000000000035'],
['80','true','8','4','136.11.86.88','cs','false','Linux','true','Europe/Vysocina','false','Chrome','95.1.1233','GeForce4 Ti 4200','aa4bbd8b2eec80be2ab2e2577429d9ab','0.5575000000000006'],
['81','false','4','2','81107141151','cs','false','Windows XP','true','Europe/Liberecky','false','Chrome','95.1.4563','GeForce4 440','a59db910d34ebb8db0d3063926795775','0.5230000000000012'],
['82','true','16','20','1,41E+11','cs','false','Windows XP','true','Europe/Liberecky','false','Chrome','95.0.4638','Master X3100 Driver','9e324db36645066db40a8c2bc3610dce','0.5500000000000013'],
['83','true','4','2','60.230.33.192','cs','true','Windows XP','true','Europe/Praha','false','Chrome','95.0.4640','Radeon 530','61d03d5d4e505eedc91936e42fe4e273','0.7335000000000007'],
['84','true','4','2','96.3.45.162','cs','false','Windows XP','true','Europe/Praha','false','Chrome','95.0.4638','Intel HD 500','45ee906eafb73445ff7560efa20d6ec4','0.5205000000000005'],
['85','false','8','4','229.66.147.115','cs','false','Windows XP','true','Europe/Praha','false','Opera?','95.0.4642','Intel 82915G Express','24c16abeff314253dc3bc9ab9de9ac7e','0.5730000000000006'],
['86','true','8','6','131.26.244.97','en-GB','false','Android 11','true','Europe/London','true','Chrome','95.0.4638','Adreno 612','8959bd046ead35c66091b680eed092e6','0.5290000000000006'],
['87','false','16','12','1,99E+11','nl','true','Android 11','true','Europe/Rotterdam','true','Chrome','95.0.4639','Mali-G52 MC2','965da38952015d49530bc62557bd2189','0.7380000000000008'],
['88','true','8','6','121.115.7.87','nl','true','Windows 7','true','Europe/Tillburg','false','Chrome','95.0.4640','radeon rx 550 2gb','f5fbe310ddfc8d11d76ba496d8c92f54','-0.4329999999999999'],
['89','true','8','6','148.58.136.85','nl','true','Windows 10','false','Europe/Arnhem','false','Chrome','95.0.4641','GeForce4 440','7b7769e5ce1a189a2ef43fe1541a60aa','0.6975000000000009'],
['90','false','4','4','104.233.61.11','nl','true','Windows 10','true','Europe/Arnhem','false','Opera?','79.3.5055','Dell 8100','941ddf901b70b12bd92bc419bdaf08fd','0.536'],
['91','true','16','12','90.56.217.77','nl','true','Windows 10','true','Europe/Assen','false','Opera?','79.6.3758','intel uhd graphics 630','8ce781c530d97f1232bd49c44e916d7d','0.5315000000000012'],
['92','true','8','6','82214126192','de','true','Windows 7','true','Europe/Munique','false','Opera?','80.0.4055','radeon rx 550 2gb','401a2a7a437a1fb91ac65b878d1aacf2','0.7575000000000014'],
['93','true','16','6','24.3.198.33','de','true','Linux','true','Europe/Dusseldorf','false','Opera?','80.0.4170','Dell 8100','69902d189b85a3f6e2eb8a4c77e86408','0.30600000000000077'],
['94','false','8','4','86.189.96.56','de','true','IOS 14','false','Europe/Stuttgart','true','Safari','60.3.5090','PowerVR 7XT GT7600','3a0c207e91e309c937ad13e34a85f88b','0.4750000000000001'],
['95','true','4','2','165.79.33.215','de','false','Windows 8','false','Europe/Munique','false','Opera?','79.3.5055','radeon rx 550 2gb','401a2a7a437a1fb91ac65b878d1aacf2','0.5710000000000012'],
['96','false','4','2','100.8.112.142','de','true','Android 11','true','Europe/Bremen','true','Chrome','95.0.4639','PowerVR GE8320','5dfd99aa08a43079781462212228aa93','0.7675000000000001'],
['97','false','4','2','100.8.112.142','de','true','Android 11','true','Europe/Bremen','true','Chrome','95.0.4639','PowerVR GE8320','5dfd99aa08a43079781462212228aa93','0.5015000000000004'],
['98','true','6','4','90.56.217.77','it','true','Android 11','false','Europe/Roma','true','Chrome','95.0.4640','Mali-G76 MP10','efff77f5ef386d962514e2c4859c3446','0.562500000000001'],
['99','false','4','2','183.57.104.170','it','false','Linux','false','Europe/Roma','false','Chrome','95.0.4641','nvidia geforce rtx 3080 founders','b0e9d90f3c1dbccc0dc8e2265098030b','0.562500000000001'],
['100','true','4','2','183.57.104.170','it','true','Windows 10','true','Europe/Napoli','false','Opera?','80.0.4170','nvidia geforce rtx 3080 founders','e4948a1f77d1f0ed9cd8320efbeda718','0.5340000000000007']]
  
  # in non-demo scenario, programmtically encode
  enc_data = [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[2,0,1,1,1,0,0,1,0,0,0,1,1,1,1,1],[3,0,0,0,2,0,0,0,0,0,0,0,0,0,2,2],[4,1,1,0,3,0,0,0,0,0,0,2,1,1,0,3],[5,1,2,2,4,0,1,1,0,0,0,1,1,1,0,4],[6,0,1,1,5,0,0,1,0,0,0,0,0,1,0,5],[7,1,0,0,6,0,0,0,0,0,0,0,1,0,3,6],[8,0,0,0,7,0,0,0,0,0,0,0,0,0,4,7],[9,1,2,3,8,0,1,1,0,0,0,1,1,1,0,8],[10,1,1,2,9,0,0,0,0,0,0,0,2,2,5,4],[11,1,1,0,10,0,0,0,0,0,0,0,3,3,6,9],[12,1,3,2,11,1,0,2,0,1,0,3,4,4,7,10],[13,0,0,2,12,1,0,2,1,1,0,2,5,5,8,11],[14,1,1,1,13,1,1,3,0,2,1,1,6,6,9,12],[15,1,3,4,14,1,0,2,0,3,0,2,7,5,10,13],[16,1,3,4,15,1,1,2,0,4,0,1,8,4,11,14],[17,0,4,4,16,1,1,0,0,3,0,1,9,7,12,15],[18,0,0,2,17,1,0,4,0,2,1,0,0,8,13,16],[19,0,1,0,18,1,1,3,0,5,1,0,10,9,14,17],[20,1,3,5,19,1,0,5,0,6,0,0,0,7,15,18],[21,1,3,5,20,1,0,5,0,7,0,0,0,10,16,19],[22,1,4,4,21,2,0,5,0,8,0,1,1,11,17,20],[23,0,0,1,22,2,0,2,0,9,0,2,2,10,18,20],[24,1,3,5,23,2,0,2,0,9,0,4,4,11,19,21],[25,1,0,2,24,2,1,3,0,10,1,0,0,12,20,22],[26,1,0,2,24,2,1,3,0,10,1,0,0,12,20,23],[27,1,0,2,24,2,1,3,0,10,1,0,0,12,20,24],[28,1,0,2,24,2,1,3,0,10,1,0,0,12,20,24],[29,1,0,1,25,2,0,1,0,11,0,1,11,13,21,24],[30,1,0,1,26,2,0,6,0,11,1,5,12,14,22,24],[31,1,0,2,27,3,1,7,0,12,0,2,13,0,23,25],[32,0,0,2,28,3,1,7,0,12,0,4,4,4,24,26],[33,0,0,1,29,3,0,0,0,13,0,1,14,0,25,27],[34,1,3,5,30,3,1,4,0,14,1,0,15,15,26,28],[35,0,0,1,29,3,0,0,0,13,0,1,14,0,25,29],[36,0,3,3,31,3,0,6,1,15,1,5,16,16,27,30],[37,0,1,0,32,3,0,3,1,15,1,0,17,8,28,29],[38,1,0,2,24,2,1,3,0,10,1,0,0,12,20,31],[39,1,0,2,24,2,1,3,0,10,1,0,0,12,20,32],[40,1,3,5,33,4,1,0,0,16,0,1,3,4,29,24],[41,1,4,5,34,4,1,2,0,16,0,0,0,17,30,24],[42,1,3,4,35,4,1,0,0,17,0,4,5,18,31,33],[43,0,4,4,36,4,1,0,0,17,0,1,18,5,32,34],[44,0,4,5,37,4,1,2,0,17,0,0,0,17,30,35],[45,1,4,6,38,4,0,3,0,18,1,2,3,19,33,36],[46,0,4,4,39,4,0,2,1,19,0,0,0,17,34,37],[47,1,4,4,40,4,0,2,0,20,0,1,3,17,34,38],[48,1,3,5,41,4,0,3,0,20,1,0,19,20,35,12],[49,0,3,5,42,5,1,5,0,21,0,1,20,18,36,39],[50,0,4,4,43,5,1,8,0,21,1,0,21,21,37,40],[51,0,4,4,44,5,1,5,0,22,0,1,22,5,32,41],[52,1,3,5,45,5,1,2,0,22,0,0,0,4,38,42],[53,1,4,4,46,5,1,2,0,22,0,1,3,5,32,43],[54,1,3,2,47,5,1,2,0,22,0,4,23,22,39,44],[55,1,4,4,48,5,0,9,0,23,1,4,24,23,40,45],[56,1,0,2,49,5,1,2,0,23,0,0,0,4,41,46],[57,1,3,5,50,5,1,0,0,23,0,4,25,4,42,47],[58,0,0,2,51,5,1,2,1,23,0,4,25,24,43,48],[59,1,0,2,52,5,1,0,0,24,0,4,25,24,44,49],[60,0,3,5,53,5,0,3,0,24,1,2,26,23,45,50],[61,1,3,3,54,5,0,10,0,24,1,5,27,25,46,51],[62,1,4,4,55,6,1,2,0,25,0,4,25,24,47,52],[63,1,4,5,56,6,1,2,0,25,0,0,0,26,48,53],[64,1,4,4,57,6,1,2,0,26,0,4,25,24,47,54],[65,1,0,1,58,6,1,0,0,26,0,2,26,17,49,55],[66,0,0,1,59,6,1,0,0,26,0,0,0,17,49,56],[67,0,3,5,60,6,1,0,0,27,0,0,0,4,29,57],[68,1,3,2,61,6,1,0,0,27,0,1,3,27,50,58],[69,1,4,5,62,6,1,2,0,28,0,0,19,28,51,59],[70,1,4,5,63,6,1,2,0,28,0,1,20,28,51,60],[71,1,0,1,64,7,1,5,0,29,0,0,0,29,52,61],[72,0,0,1,65,7,0,3,1,30,1,0,3,23,53,62],[73,1,0,2,66,7,0,6,0,31,1,5,28,30,54,63],[74,1,3,5,67,7,0,7,0,32,0,4,25,31,55,64],[75,1,3,5,68,7,0,8,0,33,1,0,26,19,56,65],[76,1,3,2,69,7,0,8,1,34,1,0,29,32,57,66],[77,1,0,1,70,7,0,1,1,34,0,0,0,33,58,67],[78,0,0,1,71,7,0,1,1,29,0,0,0,33,58,68],[79,1,3,5,72,8,1,2,0,35,0,0,30,4,59,69],[80,1,0,1,73,8,1,2,0,35,0,0,31,34,60,70],[81,0,1,0,74,8,1,5,0,36,0,0,32,26,61,71],[82,1,3,4,75,8,1,5,0,36,0,0,0,35,62,72],[83,1,1,0,76,8,0,5,0,37,0,0,19,36,63,73],[84,1,1,0,77,8,1,5,0,37,0,0,0,37,64,74],[85,0,0,1,78,8,1,5,0,37,0,4,21,33,65,75],[86,1,0,2,24,2,1,3,0,10,1,0,0,12,20,76],[87,0,3,5,79,9,0,3,0,38,1,0,3,19,66,77],[88,1,0,2,80,9,0,7,0,39,0,0,19,27,67,78],[89,1,0,2,81,9,0,0,1,40,0,0,20,26,68,79],[90,0,1,1,82,9,0,0,0,40,0,4,33,38,69,80],[91,1,3,5,83,9,0,0,0,41,0,4,34,4,29,81],[92,1,0,2,84,10,0,7,0,42,0,4,35,27,70,82],[93,1,3,2,85,10,0,2,0,43,0,4,1,38,71,83],[94,0,0,1,86,10,0,11,1,44,1,5,36,39,72,84],[95,1,1,0,87,10,1,1,1,42,0,4,33,27,70,85],[96,0,1,0,88,10,0,3,0,45,1,0,3,32,73,86],[97,0,1,0,88,10,0,3,0,45,1,0,3,32,73,87],[98,1,5,1,83,11,0,3,1,46,1,0,19,40,74,88],[99,0,1,0,89,11,1,2,1,46,0,0,20,28,75,88],[100,1,1,0,89,11,0,0,0,47,0,4,1,28,76,89]]

  print("\nDados sem tratamento: ")
  for item in raw_data:
    print(item)

  print("\nDados normalizados: ")
  for item in enc_data:
    print(item)

  m = 3  # number clusters
  seed_val = 0
  print("\nIniciando clusterizacao com %d clusters" % m)
  clustering = cluster(enc_data, m)
  print("Pronto!")

  print("\nResultado geral: ")
  print(clustering) 

  cu = cat_utility(enc_data, clustering, m)
  print("Category utility of clustering = %0.4f \n" % cu)

  print("\nDados clusterizados sem tratamento: ")
  print("=====")
  for k in range(m):
    for i in range(len(enc_data)):
      if clustering[i] == k:
        print(raw_data[i])
        print("Cluster: ", k)
    print("=====")

  print("\nFim \n")

if __name__ == "__main__":
  main()
