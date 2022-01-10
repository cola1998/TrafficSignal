import xml.dom.minidom
def get_car_number(fname):

    dom = xml.dom.minidom.parse(fname)
    root = dom.documentElement
    vehicle_list = root.getElementsByTagName('vehicle')
    NS = 0
    NSL = 0
    WE = 0
    WEL = 0
    for vehicle in vehicle_list:
        x = vehicle.getAttribute('route')
        if x == 'N_S' or x == 'S_N' or x=='S_E' or x == 'N_W':
            NS += 1
        elif x == 'S_W' or x == 'N_E':
            NSL += 1
        elif x == 'W_E' or x == 'E_W' or x=='W_S' or x == 'E_N':
            WE += 1
        elif x == 'W_N' or x == 'E_S' :
            WEL += 1

    print("NS ",NS)
    print("NSL ", NSL)
    print("WE ", WE)
    print("WEL ", WEL)

def get_car_number_4(fname,episode=-1):
    dom = xml.dom.minidom.parse(fname)
    root = dom.documentElement
    vehicle_list = root.getElementsByTagName('vehicle')
    NS = 0
    NSL = 0

    SN = 0
    SNL = 0

    WE = 0
    WEL = 0

    EW = 0
    EWL = 0
    for vehicle in vehicle_list:
        x = vehicle.getAttribute('route')
        if x == 'N_S' or  x == 'N_W':
            NS += 1
        elif x == 'S_N' or x=='S_E':
            SN += 1
        elif x == 'N_E':
            NSL += 1
        elif x =='S_W':
            SNL += 1
        elif x == 'W_E' or x=='W_S':
            WE += 1
        elif x == 'E_N' or x == 'E_W':
            EW += 1
        elif x == 'W_N' :
            WEL += 1
        elif x == 'E_S':
            EWL += 1
    if episode != -1:
        print('episode = {0}'.format(episode))
    print("NS ",NS)
    print("NSL ", NSL)
    print("SN ", SN)
    print("SNL ", SNL)
    print("WE ", WE)
    print("WEL ", WEL)
    print("EW ", EW)
    print("EWL ", EWL)
# fname = 'balance_net/high_net_3000.rou.xml'
# get_car_number_4(fname)
