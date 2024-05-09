import time

from CR_robot import CR_ROBOT, CR_bringup
from time import sleep

robot_bringup = CR_bringup('192.168.5.1', 29999)
robot_control = CR_ROBOT('192.168.5.1', 30003)

robot_bringup.ClearError()
sleep(0.5)
robot_bringup.EnableRobot()
# robot_bringup.DisableRobot()
sleep(0.5)
# robot_bringup.StartDrag()
# robot_bringup.StopDrag()

# 测试位置
# robot_control.JointMovJ(0, -35, 115, 11, -90, 40)

init_pose = [-80, -140, 550, 180, 0, 50]  # 初始位置
sentry1_pose = [-200, -160, 400, 180, 0, 50]  # 1号空间监视位
grasp1_pose = [-300, -156, 250, -165, 12, 50]  # 试管位置
work_pose = [-275, 38, 440, 180, 0, 0]  # 与人类协作位置
target_pose = [-688, -260, 400, -180, 0, 60]

way_points = [
    [-0.3295133911696346, -33.03365890127289, 113.42602097988129, 10.399606189606999, -88.44014036284223,
     39.184645460622846],
    [-0.7060385462786904, -30.939433813943822, 111.55487787723541, 9.878453629362953, -86.83848053669584,
     38.462134861098846],
    [-0.9811425633906183, -29.001394481121512, 109.76411414146423, 9.32487649598448, -85.2281241841828,
     37.79643045895636],
    [-1.2898814846011732, -26.967668900075026, 107.87789273262024, 8.773946659976378, -83.47364393682562,
     37.023333315375744],
    [-1.5482616708589032, -24.532765309553213, 106.10458755493164, 8.087928639560666, -81.67431017428133,
     36.358105523380466],
    [-1.920416320003128, -22.11505720958625, 104.10820090770721, 7.416493168807445, -79.76985128242305,
     35.619281786142565],
    [-2.3567106930453594, -19.746834960333718, 101.97332513332368, 6.90754127083697, -77.93175668888091,
     34.967598449778606],
    [-2.637110975609965, -17.244751328397534, 100.01511371135713, 6.129523418533105, -76.00726321854991,
     34.20382235602223],
    [-3.0009686882220588, -14.623629200984901, 97.95752751827241, 5.440876265680058, -74.04407061936047,
     33.598404076275216],
    [-3.509660276045295, -11.761470734589675, 95.84355819225314, 4.579611674615921, -72.11504274212825,
     32.899314211000444],
    [-3.8676824913424497, -9.19933071377405, 93.5549696683884, 3.849667405727571, -70.10575667747945,
     32.20930006013409],
    [-4.185252445046874, -6.191974493549363, 91.47017705440524, 2.8885963904327157, -68.08624317122113,
     31.398443511857103],
    [-4.500874723694705, -3.3224257528480994, 89.3718699216843, 2.1338142656003165, -66.00164310802,
     30.575975688821845],
    [-4.765862479043201, -0.33073532696246044, 87.38263928890231, 1.1719913730753073, -63.8218149681455,
     29.687152549255885],
    [-4.939802155589464, 2.8513536959650487, 85.28317773342135, 0.33613481472570017, -61.89397931495536,
     28.968074900180632],
    [-5.099965997333856, 6.119353358634751, 83.27849614620212, -0.4416985479434488, -59.780258074780775,
     28.177995297460093],
    [-5.208626667053965, 9.265735801192125, 81.08942329883578, -1.2885535515959834, -57.70925868446787,
     27.47273371294499],
    [-5.377584882228091, 12.381605912964332, 78.97775447368625, -2.2075841523719006, -55.687136396555495,
     26.717365083021946],
    [-5.5026189126765175, 15.381581208899563, 77.07576251029971, -3.109334312766552, -53.620454600854664,
     25.992611425930228],
    [-5.497023936931304, 18.595050995708274, 75.02693033218387, -4.117912283725175, -51.66472973784066,
     25.293085444941774],
    [-5.603070680066744, 21.888428192179454, 73.07046163082127, -4.97600504437369, -49.66161228652598,
     24.633721564257044],
    [-5.59660051057898, 24.783515316589465, 71.25358986854557, -5.736493778748124, -47.68109371623548,
     24.103856267708238],
    [-5.746333763166786, 27.96537991721692, 69.27682673931126, -6.427937894354914, -45.77410778610919,
     23.49663221733634],
    [-5.807011008896819, 30.42517137559615, 67.6205706596375, -7.040018317002156, -43.83330432637155,
     22.927267254364516],
    [-5.828911082346609, 33.035130601356016, 65.82565081119542, -7.68033382750809, -41.98999733330412,
     22.22768093878555],
    [-5.883958636753175, 35.98304989718816, 64.04007232189183, -8.25649974624547, -40.19423849668707,
     21.594855461024242],
    [-5.77021754647211, 38.81241301992752, 62.487529635429425, -8.744277507199923, -38.47540424269648,
     20.994799151537734],
    [-5.671724941741827, 41.52616658107346, 61.20588827133183, -9.08738268509936, -36.75111333198202,
     20.408700115294092]]


def move_to_init_pose():
    # 在初始位置
    robot_control.MovJ(init_pose[0], init_pose[1], init_pose[2],
                       init_pose[3], init_pose[4], init_pose[5])
    # robot_control.MovL(init_pose[0], init_pose[1], init_pose[2] - 100,
    #                    init_pose[3], init_pose[4], init_pose[5])
    while True:
        pose = robot_bringup.GetPose()
        if all(abs(a - b) < 5 for a, b in zip(pose[:3], init_pose[:3])):
            sleep(0.2)
            robot_bringup.ToolDOExecute(1, 0)
            robot_bringup.ToolDOExecute(2, 1)
            break
        sleep(0.1)
    sleep(0.5)


def move_to_sentry1():
    # 移动到1号空间监视位置
    robot_control.MovJ(sentry1_pose[0], sentry1_pose[1], sentry1_pose[2],
                       sentry1_pose[3], sentry1_pose[4], sentry1_pose[5])
    sleep(0.5)


def grasp_tube_to_person():
    # 移动到试管
    robot_control.MovJ(grasp1_pose[0] + 35, grasp1_pose[1], grasp1_pose[2],
                       grasp1_pose[3], grasp1_pose[4], grasp1_pose[5])
    sleep(0.5)
    robot_control.MovJ(grasp1_pose[0], grasp1_pose[1], grasp1_pose[2],
                       grasp1_pose[3], grasp1_pose[4], grasp1_pose[5])
    sleep(0.5)
    while True:
        pose = robot_bringup.GetPose()
        if all(abs(a - b) < 5 for a, b in zip(pose, grasp1_pose)):
            sleep(0.2)
            robot_bringup.ToolDOExecute(1, 1)
            robot_bringup.ToolDOExecute(2, 0)
            break
        sleep(0.1)
    sleep(0.5)
    # 移动到协作位置
    robot_control.MovJ(work_pose[0], work_pose[1], work_pose[2],
                       work_pose[3], work_pose[4], work_pose[5])
    sleep(0.5)


def place_tube():
    # 放置试管
    robot_control.MovJ(grasp1_pose[0], grasp1_pose[1], grasp1_pose[2] + 30,
                       grasp1_pose[3], grasp1_pose[4], grasp1_pose[5])
    sleep(0.2)
    robot_control.MovJ(grasp1_pose[0], grasp1_pose[1], grasp1_pose[2] + 4,
                       grasp1_pose[3], grasp1_pose[4], grasp1_pose[5])
    sleep(0.2)
    while True:
        pose = robot_bringup.GetPose()
        if all(abs(a - b) < 5 for a, b in zip(pose, grasp1_pose)):
            sleep(0.2)
            robot_bringup.ToolDOExecute(1, 0)
            robot_bringup.ToolDOExecute(2, 1)
            break
        sleep(0.1)
    sleep(0.5)


def pick_fruit():
    # robot_bringup.ToolDOExecute(1, 0)
    # robot_bringup.ToolDOExecute(2, 1)
    # sleep(1)
    robot_control.JointMovJ(-7.589085, -6.989646, 109.511246, -102.440384, -79.763664, 43.314240)
    sleep(1)
    robot_control.MovJ(-416 - 80, -105, 608, -90.109863, 43.328621, 92.571877)
    sleep(3)
    robot_bringup.ToolDOExecute(1, 1)
    robot_bringup.ToolDOExecute(2, 0)
    sleep(1)
    robot_control.MovJ(-416 - 80, -105, 608 - 100, -90.109863, 43.328621, 92.571877)
    sleep(1)
    robot_control.JointMovJ(24.377029, 46.735451, 48.623306, -66.456779, -60.104328, 24.392492)
    sleep(2)
    robot_bringup.ToolDOExecute(1, 0)
    robot_bringup.ToolDOExecute(2, 1)
    sleep(0.5)
    robot_control.JointMovJ(-7.752809, -46.523029, 113.131691, -42.857601, -85.224983, 40.318645)


def move_way_points():
    robot_control.MovJ(-200, -140, 440, -180, 0, 45)
    time.sleep(0.5)
    # for pose in way_points:
    #     robot_control.ServoJ(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5])
    #     time.sleep(0.3)


# place_tube()
move_to_init_pose()
# grasp_tube_to_person()
# move_to_sentry1()
# move_way_points()
#
# robot_bringup.SpeedL(20)
# robot_control.MovL(target_pose[0], target_pose[1], target_pose[2], target_pose[3], target_pose[4], target_pose[5])

# robot_control.MovJ(target_pose[0]/2, target_pose[1], target_pose[2]+150, target_pose[3], target_pose[4], target_pose[5])
# robot_control.MovJ(target_pose[0], target_pose[1], target_pose[2], target_pose[3], target_pose[4], target_pose[5])

robot_bringup.GetPose()
robot_bringup.GetAngle()
sleep(1)
