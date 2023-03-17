from pathlib import Path
import logging
import gdown
import argparse
import urllib.request

file_dict = {
    'IMS.7z': '1iJqTYQpHst_uYSyU5d2THsZkA8Vk6Inx',
    'FEMTOBearingDataSet.zip': '1Ab3ggc-bOdLAglHw4-a0NNtadrPJR98L',
}


def main(path_data_folder):
    """Download dataset from Google Drive."""

    logger = logging.getLogger(__name__)
    logger.info("downloading...")

    folder_raw_data = path_data_folder / "raw"
    folder_raw_data.mkdir(parents=True, exist_ok=True)
    folder_ims = folder_raw_data / "IMS"
    folder_ims.mkdir(parents=True, exist_ok=True)
    folder_femto = folder_raw_data / "FEMTO"
    folder_femto.mkdir(parents=True, exist_ok=True)

    url = "https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip"
    filename = folder_ims / "4.+Bearings.zip"
    urllib.request.urlretrieve(url, filename)

    url = "https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip"
    filename = folder_femto / "FEMTOBearingDataSet.zip"
    urllib.request.urlretrieve(url, filename)

    # # download IMS
    # gdown.download(
    #     id='1iJqTYQpHst_uYSyU5d2THsZkA8Vk6Inx', 
    #     output=str(folder_ims / "IMS.7z"), quiet=False)

    # # download FEMTO
    # gdown.download(
    #     id='1Ab3ggc-bOdLAglHw4-a0NNtadrPJR98L', 
    #     output=str(folder_femto / "FEMTOBearingDataSet.zip"), quiet=False)

    logger.info("downloading done")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser(description="Download data")

    parser.add_argument(
        "--path_data_folder",
        type=str,
        default="data/",
        help="Path to data folder that contains raw/interim/processed data folders",
    )

    args = parser.parse_args()

    path_data_folder = Path(args.path_data_folder)

    main(path_data_folder)