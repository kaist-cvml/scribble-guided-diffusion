import math

def get_all_self_attn(
        in_self_attn_list, 
        mid_self_attn_list, 
        out_self_attn_list,
        res=[]
    ):
    # Simplified dictionary initialization
    result = {key: [] for key in [8 * 8, 16 * 16, 32 * 32, 64 * 64]}
    
    # Combine all attention lists
    all_attn = [in_self_attn_list, mid_self_attn_list, out_self_attn_list]
    
    for layer_att in all_attn:
        for self_attn in layer_att:
            cur_res = self_attn.shape[1]
            if int(math.sqrt(cur_res)) in res or res == []:
                result[cur_res].append(self_attn)

    return result

def get_all_cross_attn(
        in_cross_attn_list, 
        mid_cross_attn_list, 
        out_cross_attn_list, 
        res=[16],
    ):
    if type(res) == int:
        res = [res]

    result = result = {key: [] for key in [8, 16, 32, 64]}

    all_attn = [in_cross_attn_list, mid_cross_attn_list, out_cross_attn_list]

    for layer_attn in all_attn:
        for cross_attn in layer_attn:
            cross_attn_map = cross_attn
            _, i, _ = cross_attn_map.shape
            H = int(math.sqrt(i))

            if H in res:
                result[H].append(cross_attn_map.reshape(-1, H, H, cross_attn_map.shape[-1]))

    return result
