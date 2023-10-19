# train the sdmm
python src/SDMM/train_sdmm.py

# script for MRC based on mBERT
# 1、train the PRE model in English
python src/MRC/1_train_mrc_PRE.py

# 2、run the MeTaCo-XMT with SDMM
for LANG in ar bn fi id ko ru sw te
do
  echo "meta learning on " $LANG
  python src/MRC/train_mrc.py $LANG
done

# 3、run the MeTaCo-XMT with task
for LANG in ar bn fi id ko ru sw te
do
  echo "meta learning on " $LANG
  python src/MRC/train_mrc_task.py $LANG
done

# 4、run the MeTaCo-XMT with sem
for LANG in ar bn fi id ko ru sw te
do
  echo "meta learning on " $LANG
  python src/MRC/train_mrc_sem.py $LANG
done

# 5、run the MeTaCo-XMT with random
for LANG in ar bn fi id ko ru sw te
do
  echo "meta learning on " $LANG
  python src/MRC/train_mrc_random.py $LANG
done

# script for NER based on mBERT
# 1、train the PRE model in English
python src/NER/run_tag_PRE.py --do_train --do_eval --do_predict --save_only_best_checkpoint

# 2、run the MeTaCo-XMT with SDMM
for LANG in ar bn fi id ko ru sw te zh
do
  echo "meta learning on " $LANG
  python src/NER/run_tag_syn.py --lang $LANG --do_train --do_eval --do_predict --save_only_best_checkpoint
done

# 3、run the MeTaCo-XMT with task
for LANG in ar bn fi id ko ru sw te zh
do
  echo "meta learning on " $LANG
  python src/NER/run_tag_task.py --lang $LANG --do_train --do_eval --do_predict --save_only_best_checkpoint
done

# 4、run the MeTaCo-XMT with sem
for LANG in ar bn fi id ko ru sw te zh
do
  echo "meta learning on " $LANG
  python src/NER/run_tag_sem.py --lang $LANG --do_train --do_eval --do_predict --save_only_best_checkpoint
done

# 5、run the MeTaCo-XMT with random
for LANG in ar bn fi id ko ru sw te zh
do
  echo "meta learning on " $LANG
  python src/NER/run_tag_sample.py --lang $LANG --do_train --do_eval --do_predict --save_only_best_checkpoint
done