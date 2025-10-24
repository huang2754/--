from numpy import *
import matplotlib.pyplot as plt
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])  # 修正原数组维度不一致问题
    labels = ["A", "A", "B", "B"]
    return group, labels

def file2matrix(filename):
    fr = open(filename)
    array_olines = fr.readlines()
    number_lines = len(array_olines)
    return_mat = zeros((number_lines, 3))
    class_label_vector = []
    index = 0
    for line in array_olines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector


# 数据可视化（可选执行）
file_path = r"C:\Users\Administrator\Desktop\lesson1\datingTestSet2.txt"
dating_data_mat, dating_labels = file2matrix(file_path)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2], 
           15.0*array(dating_labels), 15.0*array(dating_labels))
ax.set_xlabel('玩视频游戏所耗时间百分比')
ax.set_ylabel('每周消费的冰淇淋公升数')
ax.set_title('特征关系散点图')
# plt.show()  # 取消注释可显示散点图


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def classify0(inX, dataSet, labels, k):
    """KNN核心分类函数"""
    dataSetSize = dataSet.shape[0]
    # 计算欧氏距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat **2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances** 0.5
    # 距离排序（返回索引）
    sortedDistIndicies = distances.argsort()
    # 统计前k个近邻的标签
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 排序并返回最可能的标签
    sortedClassCount = sorted(classCount.items(), 
                             key=operator.itemgetter(1), 
                             reverse=True)
    return sortedClassCount[0][0]


def datingClassTest():
    """测试KNN分类器的准确率"""
    hoRatio = 0.5  # 测试集比例（50%数据用于测试）
    datingDataMat, datingLabels = file2matrix(file_path)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)  # 测试样本数量
    errorCount = 0.0
    
    for i in range(numTestVecs):
        # 用后50%数据作为训练集，前50%作为测试集
        classifierResult = classify0(normMat[i, :],
                                    normMat[numTestVecs:m, :],
                                    datingLabels[numTestVecs:m],
                                    3)  # k=3
        print(f"预测类别: {classifierResult}, 实际类别: {datingLabels[i]}")
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    
    print(f"\n总错误率: {errorCount / float(numTestVecs):.2%}")


def classify_person():
    """交互式输入特征，预测约会对象印象"""
    resultList = ['不喜欢', '一般', '很喜欢']  # 标签对应关系
    
    # 获取用户输入
    ffMiles = float(input("每年飞行常客里程数: "))
    percentTats = float(input("玩视频游戏时间百分比: "))
    iceCream = float(input("每周消费冰淇淋公升数: "))
    
    # 加载数据并归一化
    datingDataMat, datingLabels = file2matrix(file_path)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    
    # 处理输入数据（归一化）
    inArr = array([ffMiles, percentTats, iceCream])
    normInArr = (inArr - minVals) / ranges  # 用训练集的归一化参数处理输入
    
    # 预测结果
    classifierResult = classify0(normInArr, normMat, datingLabels, 3)
    print(f"\n你对这个人的印象可能是: {resultList[classifierResult - 1]}")


# 死循环调用交互式分类功能
if __name__ == "__main__":
    while True:
        classify_person()
        # 询问是否继续
        again = input("\n是否继续预测？(输入y继续，其他键退出): ")
        if again.lower() != 'y':
            print("程序已退出，谢谢使用！")
            break
