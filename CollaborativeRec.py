import pandas as pd
import random
import operator
from loguru import logger


class UserRecommend:
    def __init__(self):
        """
        N : 记录用户u和v交互的数量
           记录用户看过的电影数量，如： N[“1”] = 10 表示用户 ID 为 “1” 的用户看过 10 部电影；
        W : 相似矩阵，存储两个用户的相似度，如：W[“1”][“2”] = 0.66 表示用户 ID 为 “1” 的用户和用户 ID 为 “2” 的用户相似度为 0.66 ；
        train : 用户记录数据集中的数据， 格式为： train= { user : [[item1, rating1], [item2, rating2], …], …… }
        item_users : 将数据集中的数据转换为 物品_用户 的倒排表，这样做的原因是在计算用户相似度的时候，可以只计算看过相同数据集的用户之间的相似度（没看过相同数据集的用户相似度默认为 0 ），倒排表的形式为： item_users = { item : [user1, user2, …], ……}
        k : 使用最相似的 k 个用户作推荐
        n : 为用户推荐 n 个数据集
        """
        self.N = {}  # number of items user interacted, N[u] = the number of items user u interacted
        self.W = {}  # similarity of user u and user v

        self.train = {}  # train = { user : [[item1, rating1], [item2, rating2], …], …… }
        self.item_users = {}  # item_users = { item : [user1, user2, …]， …… }

        # recommend n items from the k most similar users
        self.k = 30
        self.n = 10

    def get_data(self, file_path):
        """
        @description: load data from dataset
        @file_path: path of dataset
        """
        logger.info('加载数据')
        df = pd.read_excel(file_path)
        df = df[['visitor', 'dataset_id']]
        df = df.dropna(subset=['visitor', ])
        # 每个用户访问的数据集次数是该用户对数据集的评分数
        df = df.groupby(by=['visitor', 'dataset_id'])['dataset_id'].count().reset_index(name='rating')

        # 文件太大吃光内存需要过滤一下
        df = df.sort_values(by='rating', ascending=False)
        # 取前3000个访问次数最多的
        df = df.iloc[:3000]

        datas = df.to_dict(orient='records')
        for data in datas:
            self.train.setdefault(data['visitor'], [])
            self.train[data['visitor']].append([data['dataset_id'], data['rating']])
            self.item_users.setdefault(data['dataset_id'], [])
            self.item_users[data['dataset_id']].append(data['visitor'])
        # print("self.train = ", self.train)
        # print("self.item_users = ", self.item_users)

    def similarity(self):
        """
        @description: calculate similarity between user u and user v
        计算用户u和用户v之间的相似度
        """
        logger.info(f'开始  - 计算用户u和用户v之间的相似度')
        for item, users in self.item_users.items():
            for u in users:
                self.N.setdefault(u, 0)
                self.N[u] += 1
                for v in users:
                    if u != v:
                        self.W.setdefault(u, {})
                        self.W[u].setdefault(v, 0)
                        self.W[u][v] += 1  # number of items which both user u and user v have interacted
        for u, user_cnts in self.W.items():
            for v, cnt in user_cnts.items():
                self.W[u][v] = self.W[u][v] / (self.N[u] * self.N[v]) ** 0.5  # similarity between user u and user v
        logger.info(f'结束   -   计算用户u和用户v之间的相似度')

    def recommendation(self, user):
        """
        @description: recommend items for user
        @param user : the user who is recommended, we call this user u
        @return : items recommended for user u
        为用户推荐数据集
        被推荐的用户，我们称之为u
        推荐给用户u的物品
        """
        logger.info(f'为用户推荐数据集')
        watched = [i[0] for i in self.train[user]]  # items that user have interacted
        print('watched = ', watched)
        rank = {}
        for v, similar in sorted(self.W[user].items(), key=operator.itemgetter(1), reverse=True)[
                          0:self.k]:  # order user v by similarity between user v and user u
            for item_rating in self.train[v]:  # items user v have interacted
                if item_rating[0] not in watched:  # item user hvae not interacted
                    rank.setdefault(item_rating[0], 0.)
                    rank[item_rating[0]] += similar * float(item_rating[1])
        return sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[0:self.n]


def main():
    """
    协同推荐
    :return:
    """
    userRecommend = UserRecommend()
    file_path = "visitor.xlsx"
    userRecommend.get_data(file_path)
    userRecommend.similarity()
    user = random.sample(list(userRecommend.train), 1)
    logger.info(f'user = {user}')
    # 测试
    # usse = '2b662bef-f0f2-4471-9b6c-b12330f8dcc6'
    # rec = userRecommend.recommendation(usse)
    rec = userRecommend.recommendation(user[0])

    logger.info(f'rec = {rec}')


if __name__ == '__main__':
    main()
