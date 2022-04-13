# Fine-tuning mBert for Language Identification

## Prepare Dataset

- The data used for fine-tuning is a subset of Universal Dependencies. I chosen the data entry that has the same language type as the training data used by mBert.

- Download the dataset:

  go to the ./datasets/UD20/raw directory and execute

  ```shell
  ./download_UDfiles.sh
  ```

- Preprocess the dataset:

  ```shell
  python preprocess.py --data_source UD20 --dataset_path ./datasets/UD20/raw/ud-treebanks-v2.9/ --output_path ./datasets/UD20/processed/ --eval_length 40 --force
  ```

## Fine-tuning

- Execute

  ```shell
  python main.py -config ./config/mBert.yml
  ```

- Due to the limit of time, I only provide the fine-tuned mBert model for language identification. Besides, I also try other two cross-lingual pretrained language model including XLM and XLM-R.

  XLM

  ```shell
  python main.py -config ./config/XLM.yml
  ```

  XLM-R

  ```shell
  python main.py -config ./config/XLMR.yml
  ```

## Prepare Model

- Download checkpoint from https://drive.google.com/file/d/1MF1YXCoRublDNhN9-SWZQY9ZNy4JrkVx/view?usp=sharing
- Put the checkpoint in ./ckpts/mbert/Jan-18_11-30-18/

## Identifying Language

- Input the text that you want to identify in the language_detect_demo.txt 

- And execute

  ```shell
  python language_detection.py
  ```



## Reference

- Some code for preprocessing the dataset is from https://github.com/AU-DIS/LSTM_langid

