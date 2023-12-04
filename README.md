## Reproduction of DG: Refining Generative Process with Discriminator Guidance in Score-based Diffusion Models</sub>

This repository is based on EDM https://github.com/NVlabs/edm

If want to run this repository, please refer to EDM git repo.

changed code:

```
training:   discriminator.py (written by us, core code for reproduction)
            unet.py          (copied from [ADM](https://github.com/openai/guided-diffusion), a small modification for embedding)
            fp16_util.py     (copied from [ADM](https://github.com/openai/guided-diffusion))
            logger.py        (copied from [ADM](https://github.com/openai/guided-diffusion))
            nn.py            (copied from [ADM](https://github.com/openai/guided-diffusion))

root:       generate_dg.py   (written by us, for discriminator guided sampling)
```


TODO:
1. write a classifier training code
2. do experiment
3. (optional) add other condition.



