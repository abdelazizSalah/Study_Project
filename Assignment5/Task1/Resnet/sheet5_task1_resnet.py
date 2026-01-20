from sheet5_task1_utility_tools import * 

def sheet5_task1_resnet_main(): 
    n, use_stats, p = phase1_read_arguments()

    print("Configuration:")
    print(f"  M (bytes)     : {n}")
    print(f"  Use stats     : {use_stats}")
    print(f"  p (RE mode)   : {p}")
    m = 10 # number of packets per sample

   
    training_data, validation_data, testing_data, _, validation_labels, test_labels = phase1_getting_data(n, m, use_stats=use_stats)
    print('[*] Phase 1: Prepration Done Successfully. \n --------------------------------------- \n ')
    print ('samples are: ', training_data.shape, validation_data.shape, testing_data.shape)


    if p:
        print("Using RE mode with p =", p)
    else:
        print('working on raw')


if __name__ == "__main__":
    sheet5_task1_resnet_main()
    