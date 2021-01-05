# Dataset

Introduction of several dataset

[MNIST](#MNIST)

[CelebA](#CelebA)



## MNIST

> **M**odified **N**ational **I**nstitute of **S**tandards and **T**echnology database. [Official website](http://yann.lecun.com/exdb/mnist/)



centered in a **28x28** image



Number of images:

| training set | testing set |
| :----------: | :---------: |
|    60,000    |   10,000    |

*notes: the first 5,000 testing sets are easier

Number of each category:

|  0   |  1   |  2   |  3   |  4   |  5   |  6   |  7   |  8   |  9   |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 5923 | 6742 | 5958 | 6131 | 5842 | 5421 | 5918 | 6265 | 5851 | 5949 |



## CelebA

> Large-scale CelebFaces Attributes (CelebA) Dataset

178x218

- **10,177** number of **identities**,
- **202,599** number of **face images**, and
- **5 landmark locations**, **40 binary attributes** annotations per image.



|      |             Attributes             | Positive sample | Negative sample |
| ---- | :--------------------------------: | :-------------: | :-------------: |
| 1    | 5_o_Clock_Shadow：刚长出的双颊胡须 |      22516      |     180083      |
| 2    |      Arched_Eyebrows：柳叶眉       |      54090      |     148509      |
| 3    |        Attractive：吸引人的        |     103833      |      98766      |
| 4    |       Bags_Under_Eyes：眼袋        |      41446      |     161153      |
| 5    |             Bald：秃头             |      4547       |     198052      |
| 6    |            Bangs：刘海             |      30709      |     171890      |
| 7    |          Big_Lips：大嘴唇          |      48785      |     153814      |
| 8    |          Big_Nose：大鼻子          |      47516      |     155083      |
| 9    |          Black_Hair：黑发          |      48472      |     154127      |
| 10   |          Blond_Hair：金发          |      29983      |     172616      |
| 11   |           Blurry：模糊的           |      10312      |     192287      |
| 12   |          Brown_Hair：棕发          |      41572      |     161027      |
| 13   |        Bushy_Eyebrows：浓眉        |      28803      |     173796      |
| 14   |           Chubby：圆胖的           |      11663      |     190936      |
| 15   |        Double_Chin：双下巴         |      9459       |     193140      |
| 16   |          Eyeglasses：眼镜          |      13193      |     189406      |
| 17   |          Goatee：山羊胡子          |      12716      |     189883      |
| 18   |       Gray_Hair：灰发或白发        |      8499       |     194100      |
| 19   |         Heavy_Makeup：浓妆         |      78390      |     124209      |
| 20   |      High_Cheekbones：高颧骨       |      92189      |     110410      |
| 21   |             Male：男性             |      84437      |     118162      |
| 22   | Mouth_Slightly_Open：微微张开嘴巴  |      97942      |     104657      |
| 23   |         Mustache：胡子，髭         |      8417       |     194182      |
| 24   |      Narrow_Eyes：细长的眼睛       |      23329      |     179270      |
| 25   |          No_Beard：无胡子          |     169158      |      33441      |
| 26   |       Oval_Face：椭圆形的脸        |      57567      |     145032      |
| 27   |       Pale_Skin：苍白的皮肤        |      8701       |     193898      |
| 28   |        Pointy_Nose：尖鼻子         |      56210      |     146389      |
| 29   |   Receding_Hairline：发际线后移    |      16163      |     186436      |
| 30   |      Rosy_Cheeks：红润的双颊       |      13315      |     189284      |
| 31   |        Sideburns：连鬓胡子         |      11449      |     191150      |
| 32   |           Smiling：微笑            |      97669      |     104930      |
| 33   |        Straight_Hair：直发         |      42222      |     160377      |
| 34   |          Wavy_Hair：卷发           |      64744      |     137855      |
| 35   |     Wearing_Earrings：戴着耳环     |      38276      |     164323      |
| 36   |       Wearing_Hat：戴着帽子        |      9818       |     192781      |
| 37   |     Wearing_Lipstick：涂了唇膏     |      95715      |     106884      |
| 38   |     Wearing_Necklace：戴着项链     |      24913      |     177686      |
| 39   |     Wearing_Necktie：戴着领带      |      14732      |     187867      |
| 40   |           Young：年轻人            |     156734      |      45865      |



A code to divide attributes



## Reference

[CelebA attributes name](https://zhuanlan.zhihu.com/p/35975956)

[CelebA attributes number](https://blog.csdn.net/minstyrain/article/details/83142056)