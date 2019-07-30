# Text-Generation-with-RNNs
Sequence generation using Deep Learning i.e. Recurrent Neural Nets (LSTMs) 

This repository contains PyTorch implementations of a vanilla RNN and LSTMs (Long Short-Term Memory) and a report comparing their results and attributes.

## Vanilla RNNs vs. LSTMs
First the models are compared based on a toy problem: Completing Palindromes. This enables us to compare models in terms of capturing long-term dependencies, and demonstrates that LSTMs are superior.

The current code will only print plots from previous training logs. If you delete the pickle file, the code will run for all palindrome lengths from 5 to 35. A different model is trained for every length.

## Actual Text Generation
An LSTM is trained on any data of choice; a book, a script, a WhatsApp chat log, anything. Then the model can be used to:
- Generate new text.
- Complete any given text.

The generation can be adjusted using a *temperature* variable which basically decides how greedy or random the generation is.

### Results

#### Book
Generation using a model trained on an [Agatha Christie book](_part2_text_generation/assets/book_agatha.txt). Below the model is used to complete sentences with the arbitrary choice of 60 characters.
(generated text is in *italics*, \n indicate new line)

- Sleeping beauty is*essing and staring to the boudoir who had been administered*
- Murderer was *a good said the matter of the door of the will. I was very*
- Poirot used to *much. A great specified to me to say it was the parting of*
- Of course it is *it on the fact that the prisoner had been sure of the trage*
- His mustasche was *and first interest could have been and say it?”\n\n“No, no, t*
- Oh my!*”\n\n“I don’t know what they was the cocoa, and seemed to say*
- Poirot inspected the*ther sat down and any other man who had been and strychnine*
- The crime scene is*esting the proson to me to see you are not a long to the pri*
- He said "Detective!*\n“You see that the prisoner in the mantelpiece, and the six o*
- \n\n *\n\nCHAPTER II. THE 16TH AND DRIGATE WORTES RECT AGUTE OR TH*

### Code

If you run the code [train.py](_part2_text_generation/train.py) at it's current state, it will load the previous model and perform generation.

To train a model on any file or book of your choice, preferably place the book txt under [assets](_part2_text_generation/assets), change the argument **txt_file** to the path & name of the book. Change argument **model_folder** to the model name of your choice. Model will be trained up to however many steps you specify.
Model will occasionally perform generation (sampling) during training.

#### WhatsApp Data

If you use the code with WhatsApp chat data, I would suggest uncommenting the call of the function *whatsapp_clean_data* which basically gets rid of the *Media not included* text. It's good to get rid of such text as it does not have a significant meaning in terms of dependency to words around it, and occurs often.

## Report

For any more detailed understanding of the code, please see the report.