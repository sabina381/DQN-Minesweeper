from easydict import Easydict


gridworld_size = (9, 9)
num_mine = 9

reward_dict = {'mine':-1, 'empty':1, 'overlapped':-1, 'guess':0.3, 'clear':1}
done_dict = {'mine':True, 'empty':False, 'overlapped':False, 'guess':False, 'clear':True}

color_dict = {'0':'black', '1':"skyblue", '2':'lightgreen', '3':'red', '4':'violet', '5':'brown',
                '6':'turquoise', '7':'grey', '8':'black', 'M':'white', '.':'black'}
