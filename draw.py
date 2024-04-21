import matplotlib.pyplot as plt

def draw1():
     x = [1, 5, 10, 20]
     y = [[0.2816, 0.56, 0.6563, 0.7254],
          [0.2779, 0.5806, 0.6788, 0.7497],
          [0.3036, 0.5906, 0.6948, 0.7544]]
     y = [[0.262, 0.4807, 0.5562, 0.6263],
          [0.2585, 0.4971, 0.582, 0.6512],
          [0.2758, 0.5121, 0.5938, 0.6623]]
     name = ['GUGEN/RNN', 'GUGEN/LSTM', 'GUGEN/GRU']

     plt.plot(x, y[0], label=name[0], linestyle='-', marker='o')
     plt.plot(x, y[1], label=name[1], linestyle='--', marker='s')
     plt.plot(x, y[2], label=name[2], linestyle=':', marker='^')

     # 设置图表标题和轴标签
     plt.xlabel("K")
     plt.ylabel("ACC@K")
     plt.title("Foursquare-TKY")

     plt.xlim(1, 20)
     # 设置 x 轴刻度
     plt.xticks([1, 5, 10, 15,20])

     # 添加图例
     plt.legend()

     # 显示图表
     plt.show()


def draw2():
     x = [1, 5, 10, 20]
     y = [[0.2482, 0.5725, 0.6793, 0.7395],
          [0.286, 0.5873, 0.6935, 0.763],
          [0.3036, 0.5906, 0.6948, 0.7544],
          [0.2905, 0.5743, 0.6618, 0.725],
          [0.2739, 0.5425, 0.6393, 0.6987]]
     y = [[0.2709,0.5074,0.5921,0.658],
          [0.2705,0.5065,0.5939,0.6612],
          [0.2758,0.5121,0.5938,0.6623],
          [0.2662,0.5042,0.5899,0.654],
          [0.2696,0.5067,0.5936,0.6619]]

     name = ['32', '64', '128', '256', '512']

     plt.plot(x, y[0], label=name[0], linestyle='-', marker='o')
     plt.plot(x, y[1], label=name[1], linestyle='--', marker='s')
     plt.plot(x, y[2], label=name[2], linestyle=':', marker='^')
     plt.plot(x, y[3], label=name[3], linestyle='-', marker='*')
     plt.plot(x, y[4], label=name[4], linestyle='-', marker='v')

     # 设置图表标题和轴标签
     plt.xlabel("K")
     plt.ylabel("ACC@K")
     plt.title("Foursquare-TKY")

     plt.xlim(1, 20)
     # 设置 x 轴刻度
     plt.xticks([1, 5, 10, 15, 20])

     # 添加图例
     plt.legend()

     # 显示图表
     plt.show()

def draw3():
     x = [1, 5, 10, 20]
     y = [[0.2934,0.5804,0.6701,0.7264],
          [0.2902,0.5743,0.6834,0.748],
          [0.3036,0.5906,0.6948,0.7544],
          [0.2915,0.5915,0.6882,0.7507],
          [0.2965,0.599,0.6971,0.7575]]
     y = [[0.2384,0.4629,0.543,0.6088],
          [0.2534,0.4884,0.5731,0.6424],
          [0.2758,0.5121,0.5938,0.6623],
          [0.2518,0.4806,0.5666,0.636],
          [0.2595,0.5083,0.5899,0.6562]]
     name = ['50', '100', '150', '200', '250']

     x = [50, 100, 150, 200, 250]
     y = [0.2934, 0.2902, 0.3036, 0.2915, 0.2965]
     y2 = [0.2384, 0.2534, 0.2758, 0.2518, 0.2595]
     # plt.plot(x, y[0], label=name[0], linestyle='-', marker='o')
     # plt.plot(x, y[1], label=name[1], linestyle='--', marker='s')
     # plt.plot(x, y[2], label=name[2], linestyle=':', marker='^')
     # plt.plot(x, y[3], label=name[3], linestyle='-', marker='*')
     # plt.plot(x, y[4], label=name[4], linestyle='-', marker='v')

     plt.plot(x, y, label='NYC', linestyle='-', marker='o')
     plt.plot(x, y2, label='TKY', linestyle='-', marker='*')
     # 设置图表标题和轴标签
     plt.xlabel("user dim")
     plt.ylabel("ACC@1")
     # plt.title("Foursquare-TKY")

     plt.xlim(50, 250)
     # 设置 x 轴刻度
     plt.xticks([50, 100, 150, 200, 250])

     # 添加图例
     plt.legend()

     # 显示图表
     plt.show()

def draw4():
     x = [1, 5, 10, 20]
     y_nyc = [[0.2866,0.5844,0.6916,0.7489],
          [0.3036,0.5906,0.6948,0.7544],
          [0.2863,0.5856,0.6849,0.7474],
          [0.285,0.579,0.6874,0.7472],
          [0.2979,0.5842,0.6901,0.7491]]
     y_tky = [[0.2585, 0.5019, 0.5881, 0.6536],
           [0.2758, 0.5121, 0.5938, 0.6623],
           [0.2683, 0.5064, 0.589, 0.6592],
           [0.2729, 0.5107, 0.5926, 0.6619],
           [0.2749, 0.5049, 0.5906, 0.6603]]
     x = [50, 100, 150, 200, 250]
     y = [0.2866, 0.3036, 0.2863, 0.285, 0.2979]
     y2 = [0.2585,0.2758,0.2683,0.2729,0.2749]
     name = []

     # plt.plot(x, y[0], label=name[0], linestyle='-', marker='o')
     # plt.plot(x, y[1], label=name[1], linestyle='--', marker='s')
     # plt.plot(x, y[2], label=name[2], linestyle=':', marker='^')
     # plt.plot(x, y[3], label=name[3], linestyle='-', marker='*')
     # plt.plot(x, y[4], label=name[4], linestyle='-', marker='v')

     plt.plot(x, y, label='NYC', linestyle='-', marker='o')
     plt.plot(x, y2, label='TKY', linestyle='-', marker='*')
     # 设置图表标题和轴标签
     plt.xlabel("POI dim")
     plt.ylabel("ACC@1")

     plt.xlim(50, 250)
     # 设置 x 轴刻度
     plt.xticks([50, 100, 150, 200, 250])

     # 添加图例
     plt.legend()

     # 显示图表
     plt.show()


def draw5():
     x = [1, 5, 10, 20]
     y = [[0.2981,0.5957,0.6886,0.7517],
          [0.3024,0.5904,0.6928,0.7526],
          [0.3003,0.5883,0.6817,0.7464],
          [0.3036,0.5906,0.6948,0.7544],
          [0.2915,0.5777,0.6777,0.7459]]
     y = [[0.2716,0.5119,0.6047,0.6587],
          [0.2643,0.5084,0.5908,0.6616],
          [0.254,0.492,0.5751,0.6485],
          [0.2758,0.5121,0.5938,0.6623],
          [0.2666,0.4988,0.5794,0.652]]
     name = ['50', '100', '150', '200', '250']
     x = [50, 100, 150, 200, 250]
     y = [0.2981, 0.3024, 0.3003, 0.3036, 0.2915]
     y2 = [0.2716, 0.2643, 0.254, 0.2758, 0.2666]

     # plt.plot(x, y[0], label=name[0], linestyle='-', marker='o')
     # plt.plot(x, y[1], label=name[1], linestyle='--', marker='s')
     # plt.plot(x, y[2], label=name[2], linestyle=':', marker='^')
     # plt.plot(x, y[3], label=name[3], linestyle='-', marker='*')
     # plt.plot(x, y[4], label=name[4], linestyle='-', marker='v')
     plt.plot(x, y, label='NYC', linestyle='-', marker='o')
     plt.plot(x, y2, label='TKY', linestyle='-', marker='*')

     # 设置图表标题和轴标签
     plt.xlabel("cat dim")
     plt.ylabel("ACC@1")

     plt.xlim(50, 250)
     # 设置 x 轴刻度
     plt.xticks([50, 100, 150, 200, 250])
     # 添加图例
     plt.legend()
     # plt.legend(loc='upper right')
     # 显示图表
     plt.show()

draw5()
