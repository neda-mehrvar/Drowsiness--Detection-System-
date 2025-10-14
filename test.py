import torch
import config
from pathlib import Path
import logging
import  datetime

from config import Config
from models_loss import build_model, build_loss
from Dataset_Loader import test_loader, evaluate_one_epoch


def setup_test_logging(jf):
    now = datetime.datetime.now().strftime("%d.%m.%y__%H-%M-%S")
    log_dir = Path(jf["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"test_log_file_{now}.log"
    logging.basicConfig(
        filename=str(log_filename),
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger()
    logger.info("************Test Start************")
    return logger

def main():

    jf=Config("config.json")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device*****:" , device)
    logger=setup_test_logging(jf)

    model=build_model(jf).to(device)
    criterion=build_loss(jf)

    checkpoint_path=jf["best_path"]
    checkpoint=torch.load(checkpoint_path,map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    logger.info(f"Loaded checkpoint from: {checkpoint_path}")



    test_loss,test_acc=evaluate_one_epoch(model,test_loader,criterion, device)
    logger.info(f"Test Loss: {test_loss:.4f} | Test Accuracy:{test_acc:.4f}")
    logger.info("***********Test End*************")

if __name__=="__main__":
    main()




