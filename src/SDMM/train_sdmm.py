import os
import pickle
import torch
import sys

from torch.utils.data import DataLoader, RandomSampler

sys.path.append("./src/")
import config
import data_utils
import train_helper
from tree import ud_to_list, ud_to_pos
from models import prob_wmd

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

best_dev_res = test_bm_res = test_avg_res = best_distance = 0

emb_config = {
    'mbert': 768,
    'xlm': 1280,
    'xlmr': 1024
}


def run(e):
    global best_dev_res, test_bm_res, test_avg_res, best_distance

    e.config.edim = emb_config[e.config.ml_type]

    print("*" * 25 + " DATA PREPARATION " + "*" * 25)
    if not os.path.exists("../../data/UD_data/PUD/train.pkl"):
        train_dp_1 = ud_to_list.load_conll_dataset("../../data/UD_data/PUD/positive_eng_tree.txt")
        dev_dp_1 = ud_to_list.load_conll_dataset("../../data/UD_data/PUD/positive_target_tree.txt")
        test_dp_1 = ud_to_list.load_conll_dataset("../../data/UD_data/PUD/negative_target_tree.txt")
        dp = data_utils.data_processor_pud(
            dp_eng=train_dp_1,
            dp_pos=dev_dp_1,
            dp_neg=test_dp_1,
            experiment=e)
        data = dp.process()
        output_hal = open("../../data/UD_data/PUD/train.pkl", 'wb')
        str = pickle.dumps(data, protocol=4)
        output_hal.write(str)
        output_hal.close()
    else:
        with open("../../data/UD_data/PUD/train.pkl", 'rb') as file:
            data = pickle.loads(file.read())

    print("*" * 25 + " DATA PREPARATION " + "*" * 25)
    print("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)

    model = prob_wmd(experiment=e, use_norm=True)
    #model.load('/data/home10b/wlj/code/Meta_MRC/src/Syntax_distance_model/result/eng/best.ckpt')
    #print("*" * 25 + " LOAD MODEL FROM ENGLISH " + "*" * 25)
    model.to('cpu' if not e.config.use_cuda else 'cuda:{}'.format(e.config.gpu_id))

    start_epoch = true_it = 0

    e.log.info(model)
    print("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)

    train_batch = DataLoader(data.train_data, sampler=RandomSampler(data.train_data),
                             batch_size=e.config.batch_size, drop_last=False)

    dev_batch = DataLoader(data.dev_data, sampler=RandomSampler(data.dev_data),
                             batch_size=e.config.batch_size, drop_last=False)

    test_batch = DataLoader(data.test_data, sampler=RandomSampler(data.test_data),
                            batch_size=e.config.batch_size, drop_last=False)

    # evaluator = train_helper.evaluator(model, e)

    print("Training start ...")
    # train_stats = train_helper.tracker(["loss", "STL"])
    no_new = 0
    stop_early = False

    # e.log.info("*" * 25 + " TEST EVAL: mBERT " + "*" * 25)
    best_dev_res = 999
    for epoch in range(start_epoch, e.config.n_epoch):
        if stop_early:
            break
        for it, batch in enumerate(train_batch):
            model.train()
            true_it = it + epoch * len(train_batch)
            loss, stl = model.forward(batch[0], batch[1], batch[4], batch[5], batch[8], batch[9],
                                 (batch[2], batch[3], batch[6], batch[7], batch[10], batch[11]),
                                  use_margin=0.4)

            model.optimize(loss)

            if true_it% e.config.print_every == 0 or \
                    true_it % len(train_batch) == 0:
                print("epoch: {}, it: {} (max: {}), loss: {}, stl: {}".format(epoch, it, len(train_batch), loss.item(), stl.item() if not isinstance(stl, float) else stl))

            if true_it % e.config.eval_every == 0 or \
                    true_it % len(train_batch) == 0:
                model.eval()
                # dev_dataloader = dataset.get_dev_dataloader()
                # dev_predictions = regimen.predict(probe, model, dev_dataloader)
                # reporter(dev_predictions, dev_dataloader, 'dev.txt')
                print("*" * 25 + " DEV SET EVALUATION " + "*" * 25)
                dev_mean = model.score(dev_batch)
                print(
                    "WMD: {:.7f}"
                    .format(dev_mean))
                if e.config.use_prob:
                    dev_mean1, dev_mean2 = model.pred(dev_batch)
                    print(
                        "D_Spearman: {:.7f}, U_Spearman: {:.7f}"
                        .format(dev_mean1, dev_mean2))
                    print("*" * 25 + " DEV SET EVALUATION " + "*" * 25)

                all = dev_mean
                if best_dev_res > all:
                    no_new = 0
                    best_dev_res = all

                    model.save(
                        dev_avg=best_dev_res,
                        test_avg=test_avg_res,
                        iteration=true_it,
                        epoch=epoch)
                else:
                    no_new += 1

                print("best dev.txt result: {:.7f}, "
                           .format(best_dev_res))
            if no_new == 10:
                if best_dev_res:
                    print("*" * 25 + "stop early!" + "*" * 25)
                    stop_early = True
                    break
                else:
                    no_new = 0
            it += 1

        model.save(
            dev_avg=best_dev_res,
            test_avg=test_avg_res,
            iteration=true_it,
            epoch=epoch + 1,
            name="latest")
    model.eval()
    print("*" * 25 + " DEV SET EVALUATION " + "*" * 25)
    dev_mean1 = model.score(dev_batch)
    print(
        "WMD: {:.7f}"
        .format(dev_mean1))
    print("*" * 25 + " TEST SET EVALUATION " + "*" * 25)
    dev_mean1, (pos_accuracy, neg_accuracy, all_accuracy) = model.pred_s(test_batch)
    print(
        "test WMD: {:.7f}, pos_accuracy: {:.7f}, neg_accuracy: {:.7f}, all_accuracy: {:.7f}"
        .format(dev_mean1, pos_accuracy, neg_accuracy, all_accuracy))

    print("*" * 25 + " TEST SET EVALUATION " + "*" * 25)
    dev_mean1, (pos_accuracy, neg_accuracy, all_accuracy) = model.pred_mbert(test_batch)
    print(
        "mbert test WMD: {:.7f}, pos_accuracy: {:.7f}, neg_accuracy: {:.7f}, all_accuracy: {:.7f}"
        .format(dev_mean1, pos_accuracy, neg_accuracy, all_accuracy))

    if e.config.use_prob:
        dev_mean1, dev_mean2 = model.pred(dev_batch)
        print(
            "D_Spearman: {:.7f}, U_Spearman: {:.7f}"
            .format(dev_mean1, dev_mean2))
        print("*" * 25 + " DEV SET EVALUATION " + "*" * 25)


if __name__ == '__main__':
    args = config.get_base_parser().parse_args()
    args.use_cuda = torch.cuda.is_available()
    if args.use_cuda:
        args.gpu_id = 0

    def exit_handler(*args):
        print(args)
        print("best dev.txt result: {:.7f}, "
              "STSBenchmark result: {:.7f}, "
              "test average result: {:.7f}"
              .format(best_dev_res, test_bm_res, test_avg_res))
        exit()

    train_helper.register_exit_handler(exit_handler)

    with train_helper.experiment(args, args.save_file_path) as e:
        e.log.info("*" * 25 + " ARGS " + "*" * 25)
        e.log.info(args)
        e.log.info("*" * 25 + " ARGS " + "*" * 25)

        # run_eng(e)
        run(e)