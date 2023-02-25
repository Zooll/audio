# import torchaudio

# waveform, sample_rate = torchaudio.load('audio/foo.wav')  # load tensor from file
# torchaudio.save('foo_save.wav', waveform, sample_rate)  # save tensor to file

import pprint
pp = pprint.PrettyPrinter(indent=4)
import torch
import pickle
from pytorch_lightning import LightningDataModule
from torchaudio.datasets.librispeech import LIBRISPEECH

from examples.self_supervised_learning.data_modules._utils import BucketizeBatchSampler, CollateFnWav2Vec2, DistributedBatchSampler, HuBERTDataSet
from torchaudio.prototype.models._conformer_wav2vec2 import * 


librispeech_cls = LIBRISPEECH
dataset = librispeech_cls("", url="train-clean-100", download=False)
pp.pprint(dataset[0])
pp.pprint(dataset[0][0].shape)

pp.pprint(dataset[1])
pp.pprint(dataset[1][0].shape)

print(dataset[0][0].size(1))
# len_list = [d[0].size(1) for d in dataset]
# with open('len_list.obj', 'wb') as fp:
#     pickle.dump(len_list, fp)
    
len_list = {}
with open('len_list.obj', 'rb') as fp:
    len_list = pickle.load(fp)

sampler = BucketizeBatchSampler(
    len_list,
    num_buckets=10000,
    max_token_count=30 * 16000,
    min_len=32000,
    max_len=250000,
    shuffle=True,
)

dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=CollateFnWav2Vec2(pad=True, rand_crop=False),
        )

for k in dataloader:
    print(k)
    break


model = conformer_wav2vec2_base()
