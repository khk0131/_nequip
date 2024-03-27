from e3nn import o3

def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False