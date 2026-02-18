from training.train_GGNN import train_GGNN
from testing.test_GGNN import run_GGNN_test
from utility.utils_evaluation import load_GGNN

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ====== PATH ======
    inp_path  = r"//home/zagaria/Tesi/Tesi/Networks-found/20x20_branched_copy_copy_copy.inp"
    ggnn_ckpt = r"/home/zagaria/Tesi/Tesi/wdn-CW-leak-localization/saved_models/ggnn_model.pt"

    # ====== TRAINING HYPERPARAMETERS ======
    NUM_STEPS = 50
    EPOCHS = 100
    LR = 2e-1
    LEAK_AREA = 0.1

    HIDDEN_SIZE = 132
    PROPAG_STEPS = 6

    # ====== TEST PARAMETERS ======
    NUM_TEST = 100
    TOP_K = 5

    print("\n================= TRAIN GGNN =================")
    train_GGNN(
        inp_path=inp_path,
        EPOCHS=EPOCHS,
        num_steps=NUM_STEPS,
        LR=LR,
        LEAK_AREA=LEAK_AREA,
        HIDDEN_SIZE=HIDDEN_SIZE,
        PROPAG_STEPS=PROPAG_STEPS
    )

    print("\n================= LOAD MODEL =================")
    model = load_GGNN(ggnn_ckpt)

    print("\n================= TEST GGNN =================")
    run_GGNN_test(
        inp_path=inp_path,
        model=model,
        num_test=NUM_TEST,
        num_steps=NUM_STEPS,
        K=TOP_K
    )
