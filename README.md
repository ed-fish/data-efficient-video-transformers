# stacked video transformer

  An experiment in stacking transformer networks with convolutional token inputs. 
  
  Videos are broken down into frames, clips, scenes, and videos with a transformer network aggregating representations at each stage via a learnable CLS token. 
  
  The network learns the representation of a clip via attention on frames, a scene via attention on clips, and a whole video  via attention of scenes. 
  This reduces the computational complexity of implementing transformers on very long videos as each temporal level is encoded into a single CLS token.
  
  I am still experimenting with this method and currently working through the engineering challenges before I can fully explore the through a comprehensive research project and paper. 
  
  This work is linked to my current work on efficient multi-modal transformer networks for long videos but is a different approach. 
  
  ***This code will not run correctly without the formatted dataset. I will endeavor to release a generic version for MIT and Kinetics asap***
