
def macs_cal(depth, channels, token_len):
    return depth * (24 * token_len * channels * channels + 4 * token_len * token_len * channels)

if __name__ == '__main__':
    print(macs_cal(12, 768, 320) * 1e-9)