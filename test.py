from framework.mysql import *
from framework.common import *
import pymysql
import pandas as pd
import numpy as np
import json
import datetime


class Process:
    def __init__(self, homeid, home_mac, home_device, command_list, mac_mid_type):
        self.homeid = homeid
        self.home_mac = pd.DataFrame(home_mac)
        self.home_device = pd.DataFrame(home_device)
        self.command_list = command_list
        self.mac_mid_type = pd.DataFrame(mac_mid_type)
        self.logger = get_logger(logging.getLogger(__name__))

    # 根据homeid查询家庭设备
    def get_facilities(self):
        # 设备mac列表
        mac_list = self.home_mac[self.home_mac['mac'] == self.homeid]
        if len(mac_list) == 0:
            self.logger.info('家庭中没有设备')
            return
        # 设备mid, roomid
        mac_mid_roomid = []
        for i in range(len(mac_list)):
            if mac_list[i] in self.home_device['mac'].tolist():
                idx = np.where(self.home_device['mac'].tolist() == mac_list)
                mid = self.home_device.iloc[idx, -1]
                roomid = self.home_device.iloc[idx, -2]
                mac_mid_roomid.append([mac_list[i], mid, roomid])
            else:
                continue
        if len(mac_mid_roomid) == 0:
            self.logger.info('家庭中有设备，但是没有查到对应的mid')
            return

        return mac_mid_roomid, mac_list

    # 查询场景
    def select_scene(self):
        conn, cursor = get_connection_mysql()
        cursor.execute('select * from table')
        scene_list = cursor.fetchall()[0]
        cursor.close()
        conn.close()
        return scene_list

    # 场景过滤
    def filter_scene(self):
        conn, cursor = get_connection_mysql()
        mac_mid_roomid, mac_list = Process.get_facilities(self)
        if mac_mid_roomid is None:
            self.logger.info('家庭中设备信息不全，无法推荐场景')
            return
        scene_list = Process.select_scene(self)
        satisfy_scene_idx = []
        for i in range(len(scene_list)):
            # 判断家庭里的设备是否满足场景
            # 符合的场景添加到列表,并保存场景对应的mid
            # 先解析字段，找出所需场景的mid
            scene = json.loads(scene_list[i])
            mid_list_condition = []
            mid_list_action = []
            condition = scene['Term']['condition']
            action = scene['Term']['action']
            if len(condition) != 0:
                for j in range(len(condition)):
                    one_condition = condition[j]
                    id = one_condition['id']
                    # 这里需要修改表名
                    des_list = get_des(id)
                    if des_list[1] == 'dev':
                        mid_list_condition.append([j, des_list[2]])
                    else:
                        continue
            if len(action) != 0:
                for k in range(len(action)):
                    one_action = action[k]
                    id = one_action['id']
                    # 需要修改表名
                    des_list = get_des(id)
                    if des_list[1] == 'dev':
                        mid_list_action.append([k, des_list[2]])

            home_mid_list = [example[1] for example in mac_mid_roomid]
            # 如果条件类要求的设备，家庭中只要有一台没有，就不符合
            if not (set([example[1] for example in mid_list_condition]) <= set(home_mid_list)):
                self.logger.info('家庭中设备不满足条件类设备，过滤掉')
                continue
            else:
                if len([example[1] for example in mid_list_action]) == 0:
                    self.logger.info('家庭中动作执行不包括设备执行类， 可以推荐')
                    satisfy_scene_idx.append([i, mid_list_condition, mid_list_action])
                else:
                    if len(set(home_mid_list) & set([example[1] for example in mid_list_action])) == 0:
                        self.logger.info('家庭内没有可执行动作类设备,过滤掉')
                    else:
                        self.logger.info('有可执行动作的设备，可以推荐')
                        satisfy_scene_idx.append([i, mid_list_condition, mid_list_action])

        return satisfy_scene_idx

    def sort_scene(self, satisfy_scene_idx):

        # sort_list = ['ariCon', 'Fan', 'dehumidifier']
        scene_type = []
        origin_index = []
        for i in range(len(satisfy_scene_idx)):
            origin_index.append(satisfy_scene_idx[0])
            mid_list = satisfy_scene_idx[1] + satisfy_scene_idx[2]
            type_list = []
            for mid in mid_list:
                mid_type = self.mac_mid_type[self.mac_mid_type['mid'] == mid, -1]
                type_list.append(mid_type)

            scene_type.append(list(set(type_list)))

        # # 按设备多少排序
        # scene_type_num_list = [len(type_list) for type_list in scene_type]
        # sorted_num_idx = list(np.array(scene_type_num_list).argsort())
        #
        # new_idx = []
        # for idx in sorted_num_idx:
        #     new_idx.append(origin_index[idx])
        #
        # new_idx = new_idx[::-1]
        #
        # new_scene_type = []
        # for idx in sorted_num_idx:
        #     new_scene_type.append(scene_type[idx])
        #
        # new_scene_type = new_scene_type[::-1]

        # 在按照设备类型排序
        scene_type_dict = {}
        for i in range(len(scene_type)):
            type_list = scene_type[i]
            example = []
            if len(type_list) == 0:
                example = [0, 0, 0]
            else:
                for type_one in type_list:
                    if type_one == 'airCon':
                        example.append(1)
                    else:
                        example.append(0)
                    if type_one == 'Fan':
                        example.append(1)
                    else:
                        example.append(0)
                    if type_one == 'dehumidifier':
                        example.append(1)
                    else:
                        example.append(0)
            scene_type_dict[origin_index[i]] = example

        pd_scene_type = pd.DataFrame(scene_type_dict)
        type_num = []
        for i in range(len(pd_scene_type)):
            type_num.append(sum(pd_scene_type.iloc[i, :].values))
        pd_scene_type['type_num'] = type_num
        sorted_pd_scene_type = pd_scene_type.sort_values(by='type_num', ascending=False).reset_index()
        label_list = []
        for i in range(len(sorted_pd_scene_type)):
            if sorted_pd_scene_type.iloc[i, 1] == 1:
                label = 'airCon'
            elif sorted_pd_scene_type.iloc[i, 2] == 1:
                label = 'fan'
            elif sorted_pd_scene_type.iloc[i, 3] == 1:
                label = 'dehumidifier'
            else:
                label = 'none'
            label_list.append(label)

        sorted_pd_scene_type['label'] = label_list

        # 按照设备多少进行分组
        group_by_num = []
        num_ = np.unique(type_num)
        for num in num_:
            group_by_num.append(sorted_pd_scene_type[sorted_pd_scene_type['type_num'] == num])

        result_idx = []
        for i in range(len(group_by_num), -1, -1):
            if len(group_by_num[i]) <= 2:
                result_idx.extend(group_by_num[i]['index'].tolist())
            else:
                group_fetch = []
                group_type = np.unique(group_by_num[i][group_by_num[i]['label']])
                if 'airCon' in group_type:
                    airCon_ = group_by_num[i][group_by_num[i]['label'] == 'airCon']
                    group_fetch.append(airCon_['index'].tolist())
                elif 'fan' in group_type:
                    fan_ = group_by_num[i][group_by_num[i]['label'] == 'fan']
                    group_fetch.append(fan_['index'].tolist())
                elif 'dehumidifier' in group_type:
                    dehumidifier_ = group_by_num[i][group_by_num[i]['label'] == 'dehumidifier']
                    group_fetch.append(dehumidifier_['index'].tolist())
                else:
                    none = group_by_num[i][group_by_num[i]['label'] == 'none']
                    group_fetch.append(none['index'].tolist())
                result_idx.extend(group_fetch)




    def insert_result(self):
        conn, cursor = get_connection_mysql()
        satisfy_scene_idx = Process.filter_scene(self)

        command = {'precondition': {
            'items': [{'data': {'timing': "08:00~23:59", 'weeks': ['1', '2', '3', '4', '5']}, 'type': 0},
                      {'data': {'actionTips': '房间有人', 'id': ['GREE', ''], 'properties': {'RoomPeopleNum': "3@#$0"}},
                       'type': 30},
                      {'data': {'cityName': '珠海市', 'qlty': '中度污染'}, 'type': 24},
                      ], 'mode': 0},
                   'action': [
                       {'task': {'cmd': '', 'dat': [], 'delayTime': 0, 'homeId': 0, 'key': '', 'mac': '', 'musicId': 0},
                        'type': 0, 'taskNum': 0}]}

        # 查询满足条件的场景的指令，并给指令mac、homeId赋值
        recommend_scene_command = []
        recommend_scene_idx = []
        for i in range(len(satisfy_scene_idx)):
            # 符合的场景索引
            scene_idx = satisfy_scene_idx[i][0]
            recommend_scene_idx.append(scene_idx)
            # 每个场景下的设备mid
            scene_mid_list = satisfy_scene_idx[i][1]
            scene_command = self.command_list[scene_idx]
            if len(scene_mid_list) == 1:
                scene_mac = self.home_device[self.home_device['mid'] == scene_mid_list[0]]
                scene_command = scene_command + scene_mac + scene_mid

            else:
                # 可能有多个设备的action指令，一个一个添加mid
                for j in range(len(scene_mid_list)):
                    scene_mac = self.home_device[self.home_device['mid'] == scene_mid_list[j]]
                    # 分别对场景指令中的多个action添加mac、homeid
                    scene_command = scene_command + scene_mac + scene_mid

            recommend_scene_command.append(scene_command)

        ctime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sql = 'insert into table values (0, %s, %s, %s, %s)'

        cursor.execute(sql, (ctime, self.homeid, recommend_scene_idx, json.dumps(json.dumps(recommend_scene_command))))
        conn.commit()
        conn.close()


def get_connection_mysql():
    host = config.get('mysql', 'host')
    port = config.get('mysql', 'port')
    user = config.get('mysql', 'user')
    password = config.get('mysql', 'password')
    db = config.get('mysql', 'database')

    conn = pymysql.connect(host=host, port=int(port), user=user, password=password, database=db, charset='utf8')
    cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
    return conn, cursor


def get_des(id):
    conn, cursor = get_connection_mysql()
    cursor.execute('select Des from table2 where id=%s' % id)
    des = cursor.fetchall()[0]['Des']
    des_list = des.split('_')

    return des_list
