import argparse
import sys
import os

def main():
    print("os.path.abspath(__file__)", os.path.abspath(__file__))
    script_folder = os.path.abspath(__file__)[:-7]
    os.system("docker build -t ares:latest " + script_folder)

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_folder", type=str, required=False, default="src/")
    parser.add_argument("-d", "--data_folder", type=str, required=False, default="data/")
    parser.add_argument("-desc", "--descriptor", type=str, required=False, choices=['sift', 'surf', 'orb'], default='surf')
    parser.add_argument('-e','--extra_desc_param', type=int, required=False, default=2500)
    parser.add_argument('-c', '--calibration', dest='do_calibration', action='store_true')
    parser.set_defaults(do_calibration=False)
    opt = parser.parse_args()


    split_param = opt.src_folder.rsplit('/')
    last_folder_name = split_param[len(split_param)-2] if len(split_param[len(split_param)-1]) == 0 else split_param[len(split_param)-3]
    src_folder_path = os.path.dirname(os.path.abspath(opt.src_folder)) + '/' + last_folder_name + '/'
    
    split_param = opt.data_folder.rsplit('/')
    last_folder_name = split_param[len(split_param)-2] if len(split_param[len(split_param)-1]) == 0 else split_param[len(split_param)-3]
    data_folder_path = os.path.dirname(os.path.abspath(opt.data_folder)) + '/' + last_folder_name + '/'
    
    do_calibration_string = "--calibration" if opt.do_calibration else ""
    print("opt.do_calibration", opt.do_calibration)
    print("do_calibration_string", do_calibration_string)
    
    print(src_folder_path)
    print(data_folder_path)

    os.system("xhost +")
    cmd = "docker run "
    cmd += "-v " + src_folder_path + ":/src/ "
    cmd += "-v " + data_folder_path + ":/data/ "
    cmd += "--rm -it -e DISPLAY=unix$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix "
    cmd += "ares:latest"
    cmd += " python3 src/main.py"
    cmd += " --data_folder " + opt.data_folder
    cmd += " --descriptor " + opt.descriptor
    cmd += " --extra_desc_param " + str(opt.extra_desc_param)
    cmd += " " + do_calibration_string
    print("Running command line...\n" + cmd)
    os.system(cmd)






if __name__ == "__main__":
    main()
