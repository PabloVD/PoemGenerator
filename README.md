# PoemGenerator

Train a recurrent neural network with the corpus of a given author to generate poems imitating his/her style.

It makes use of a LSTM architecture implemented in [pytorch](https://pytorch.org/), trained with a compilation of poems by a writer. Given the relatively short corpus that are used, character tokenization is employed.

The network is able to replicate the style, the metre (if a regular corpus is employed, e.g., sonnets), and also sometimes assonant rhymes. It is a simple generator though, it may show nonexistent words, and verses with little semantical sense.

Based on a tutorial from the Udacity course [Intro to Deep Learning with PyTorch](https://classroom.udacity.com/courses/ud188/).

## Usage

Here is a brief description of the codes included:

- `samples.ipynb`: jupyter notebook to generate sample poems with pretrained models.

- `main.py`: main code to train the network and generate samples.

- `preprocess_texts.py`: compiles and preprocess the texts of an author.

In the folder `source`, the architecture, parameters and several routines are defined.

## Poets and sources

The network is trained with the corpus of poems of several authors writing in different languages and styles. Those currently included are:

- [Lope de Vega](https://en.wikipedia.org/wiki/Lope_de_Vega) (1562-1635, spanish, baroque). Complete set of sonnets, extracted from [Cervantes Virtual](http://www.cervantesvirtual.com/obra-visor/sonetos--34/html/ffe58ca0-82b1-11df-acc7-002185ce6064_4.html#I_0_).

- [Jacint Verdaguer](https://en.wikipedia.org/wiki/Jacint_Verdaguer) (1845-1902, catalan, romanticism/*renaixença*). Several long poems, with different metres, extracted from [Wikisource](https://ca.wikisource.org/wiki/Autor:Jacint_Verdaguer_i_Santal%C3%B3).

- [John Donne](https://en.wikipedia.org/wiki/John_Donne) (1572-1631, english, metaphysical). Compilation *The Poems of John Donne, Volume 1*, with different metres, from [Project Gutenberg](https://gutenberg.org/ebooks/48688).

## Samples

Some samples of generated texts by the network. An initial verse provided by the user is required.

* Lope de Vega, with the initial verse *versos de amor, conceptos esparcidos,*:

  >*versos de amor, conceptos esparcidos,*
  >
  >*por las espaldas del pintor del miedo.*
  >
  >*para escarando el mar, el rey el día,*
  >
  >*divina mano, en que sustenta amiga,*
  >
  >*y así con el camino de mis ojos*
  >
  >*de sangre dios el sol en los cigeros.*

* Jacint Verdaguer, with the initial verse *entre'ls arbres de l'illa delitosa*:

  > *entre'ls arbres de l'illa delitosa,*
  >
  >*com un plor del mirade ab la terra estelada,*
  >
  >*desde 'l camí de l'aygua de llur capella somniosa,*
  >
  >*l'altre l' amplí dels cortinatges y deus.*

* John Donne, with the initial verse *no man is an island*:

  >*no man is an island.*
  >
  >*if i have more then his actions by all things,*
  >
  >*or sense, and shall see then before.*
  >
  >*if then it seem'd a soule of god,*
  >
  >*and seest a mortine to be such a part*
  >
  >*with mania line, being so from her last see.*

## Contact

For comments, questions etc. you can reach me at <pablo.villanueva.domingo@gmail.com>
