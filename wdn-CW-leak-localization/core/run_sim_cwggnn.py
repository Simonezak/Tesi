from training.train_CWGGNN import train_CWGGNN
from testing.test_CWGGNN import run_CWGGNN_test
from utility.utils_evaluation import load_CWGGNN

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ====== PATH ======
    inp_path = r"/home/zagaria/Tesi/Tesi/Networks-found-final/modena_BSD.inp"
    inp_path2 = r"/home/zagaria/Tesi/Tesi/Networks-found-final/modena_BSD.inp"
    model_ckpt = r"/home/zagaria/Tesi/Tesi/wdn-CW-leak-localization/saved_models/cw_ggnn_Modena.pt"

    # ====== TRAINING HYPERPARAMETERS ======
    NUM_STEPS = 50
    EPOCHS = 150
    
    LR = 1e-2
    LEAK_AREA = 0.01

    HIDDEN_SIZE = 132
    PROPAG_STEPS = 6
    TOPO_MLP_HIDDEN = 32
    MAX_CYCLE_LENGTH = 8

    # ====== TEST PARAMETERS ======
    NUM_TEST = 30
    TOP_K = 5

    print("\n================= TRAIN CW-GGNN =================")
    train_CWGGNN(
        inp_path=inp_path,
        EPOCHS=EPOCHS,
        num_steps=NUM_STEPS,
        LR=LR,
        LEAK_AREA=LEAK_AREA,
        HIDDEN_SIZE=HIDDEN_SIZE,
        PROPAG_STEPS=PROPAG_STEPS,
        TOPO_MLP_HIDDEN=TOPO_MLP_HIDDEN,
        MAX_CYCLE_LENGTH=MAX_CYCLE_LENGTH
    )

    print("\n================= LOAD MODEL =================")
    model = load_CWGGNN(model_ckpt,inp_path)

    print("\n================= TEST CW-GGNN =================")
    run_CWGGNN_test(
        inp_path=inp_path2,
        model=model,
        num_test=NUM_TEST,
        num_steps=NUM_STEPS,
        K=TOP_K
    )
