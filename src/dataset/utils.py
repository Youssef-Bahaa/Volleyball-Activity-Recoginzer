
categories_dct = {
    'l-pass': 0,
    'r-pass': 1,
    'l-spike': 2,
    'r_spike': 3,
    'l_set': 4,
    'r_set': 5,
    'l_winpoint': 6,
    'r_winpoint': 7
}

def activity2id(activity):
    return categories_dct[activity]

def id2activity(id):
    for action , id_ in categories_dct.items():
        if id_ == id:
            return action
    return ''
