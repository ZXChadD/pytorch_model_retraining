import re
def main():
    file = open("trainval.txt", "w+")
    for i in range(450):
      file.write(str(i) + '\n')


    # files = ['../saved_models/mb2-ssd-lite-Epoch-0-Loss-12.08188806261335.pth', '../saved_models/mb2-ssd-lite-Epoch-0-Loss-12.08894647870745.pth', '../saved_models/mb2-ssd-lite-Epoch-10-Loss-4.292270864759173.pth', '../saved_models/mb2-ssd-lite-Epoch-20-Loss-3.561148064477103.pth', '../saved_models/mb2-ssd-lite-Epoch-30-Loss-3.3576602254595076.pth', '../saved_models/mb2-ssd-lite-Epoch-40-Loss-2.8613594940730502.pth', '../saved_models/mb2-ssd-lite-Epoch-50-Loss-2.975963285991124.pth', '../saved_models/mb2-ssd-lite-Epoch-60-Loss-2.553285837173462.pth', '../saved_models/mb2-ssd-lite-Epoch-70-Loss-2.673589059284755.pth', '../saved_models/mb2-ssd-lite-Epoch-80-Loss-2.3622590814317976.pth', '../saved_models/mb2-ssd-lite-Epoch-90-Loss-2.237060683114188.pth', '../saved_models/mb2-ssd-lite-Epoch-100-Loss-2.1904413359505788.pth', '../saved_models/mb2-ssd-lite-Epoch-110-Loss-2.182618124144418.pth', '../saved_models/mb2-ssd-lite-Epoch-120-Loss-2.184490850993565.pth', '../saved_models/mb2-ssd-lite-Epoch-130-Loss-2.162746940340315.pth', '../saved_models/mb2-ssd-lite-Epoch-140-Loss-2.1616919381277904.pth', '../saved_models/mb2-ssd-lite-Epoch-150-Loss-2.296015671321324.pth', '../saved_models/mb2-ssd-lite-Epoch-160-Loss-2.2162062100001743.pth', '../saved_models/mb2-ssd-lite-Epoch-170-Loss-2.4468394347599576.pth', '../saved_models/mb2-ssd-lite-Epoch-180-Loss-2.29133141040802.pth', '../saved_models/mb2-ssd-lite-Epoch-190-Loss-2.34732495035444.pth', '../saved_models/mb2-ssd-lite-Epoch-200-Loss-2.6815099716186523.pth']
    # all_files = []
    # for file in files:
    #     current_file = re.findall('[^-]+(?=.pth)', file)
    #     all_files.append(float(current_file[0]))
    #
    # min_file = min(all_files)
    #
    # for x in files:
    #     if str(min_file) in x:
    #         min_file = x

if __name__ == '__main__':
    main()