
import time
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="filter CSI_DATA from console_test logfile")
    parser.add_argument('-S', '--src', dest='src_file', action='store', required=True,
                        help="console_test logfile")
    parser.add_argument('-G', '--gen', dest='gen_file', action='store', default='gen.txt',
                        help="output file saved csi data")                        

    args = parser.parse_args()
    print(args)

    f_src = open(args.src_file, 'r')
    lines = f_src.readlines()

    f_gen = open(args.gen_file, 'w+')
    for line in lines:
        print(line)
        f_gen.write(line)
        time.sleep(0.01)

    f_gen.close()    
    f_src.close()