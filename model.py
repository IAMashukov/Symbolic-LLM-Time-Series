import logging 
from dataclasses import dataclass 
from typing import Any, Dict, List, \
                Literal, Optional, Tuple, Union


import torch
import torch.nn as nn 

from transformers import (
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    GenerationConfig,
    PreTrainedModel,
)

import model 



logger = logging.getLogger(__file__)


@dataclass
class ModelConfig:
    
    # TODO

    """
    This cass holds all the configuration parameters to be used 
    by ``ModelTokenizer`` and ``Model``.
    """
    device: Union[str, torch.device] = 'cuda'
    tokenizer_class: str 
    tokenizer_kwargs: Dict[str, Any]
    context_length: int 
    n_classes: int
    # prediction_length: int 
    n_tokens: int 
    n_special_tokens: int 
    pad_token_id: int 
    eos_token_id: int 
    use_eos_token: bool 
    model_type: Literal["encoderT5", "seq2seq"]
    # num_samples: int 
    # temperature: float 
    # top_k: int 
    # top_p: float

    def __post_init__(self):
        assert(
            self.pad_token_id < self.n_special_tokens 
            and self.eos_token_id < self.n_special_tokens 
        ), f"Special token ids must be smaller than {self.n_special_tokens}"

    def create_tokenizer(self) -> "ModelTokenizer":
        
        class_ = getattr(model, self.tokenizer_class)
        return class_(**self.tokenizer_kwargs, config = self)
    



class ModelTokenizer:

    """
    This will definend how time series are mapped into token IDs and back.

    This specific class is a base abstract class that defines the interface 
    and expected methods for all tokenizers to be used. 

    Mimicking Chronos, this will be designed to discretize time series into tokens - 
    so this class describes how that should be done:

    From real-vaued time series -> token IDs

    But unlike Chronos, we will not be mapping token IDs back to real valued functions - 
    we are performing classification.
    """

    def context_input_transform(
            self,
            context: torch.Tensor
    ) -> torch.Tensor:
        """
        Turn a batch of time series into token IDs, 

        Parameters
        ----------

        context
            A tensor shaped (batch_size, num_steps), containing the timeseries to classify.
            Note that until we have come up with an efficient way to tokenize multivariate time series, 
            we are prototyping to map flattened time series, hence the shape.
        
            
        Returns
        -------

        token_ids
            A tensor of integers, shaped (batch_size, num_steps + 1)
            if ``config.use_eos_token`` and (batch_size, num_steps)
            otherwise, containing token IDs for the input series.
        
        """
        raise NotImplementedError()
    



class MeanScaleUniformBins(ModelTokenizer):

    def __init__(self,
                low_limit: float,
                high_limit: float, 
                config: ModelConfig) -> None:
        """
        | Parameter    | Meaning                                                |
        | ------------ | ------------------------------------------------------ |
        | `low_limit`  | Minimum value the tokenizer expects (e.g., `-15.0`)    |
        | `high_limit` | Maximum value the tokenizer expects (e.g., `+15.0`)    |
        | `config`     | Full ChronosConfig object, containing `n_tokens`, etc. |

        This will let the tokenizer quantize real values 
        between low_limit and high_limit into uniform bins.

        The flow will be:
        
        1. At input time:
            - Normalize if needed
            - Use torch.bucketize() or similar to assign token IDs
        
        2. Save metadata like bin centers (needed for decoding)
        
        3. At output time:
            Map token ID back to corresponding 
            bin center (to recover approximate value).
        """
        # Stores the configuration object so that 
        # other methods can access it 
        # (e.g., context_length, n_tokens, use_eos_token, etc.).
        self.config = config 


        # Binning setup:
        """
        torch.linspace(a,b,n) creates n evenly spaced values between a and b.

        These are the centers of the bins that values will be quantized to. 

        
        Let's say:

        n_tokens = 516 
        n_special_tokens = 4 
        Then the number of data bins = 511 

        So this will make 511 evenly spaced bin centers 
        between low_limit and high_limit.

        | Value                  | Meaning                              |
        | ---------------------- | ------------------------------------ |
        | `n_tokens = 516`       | Total size of vocabulary             |
        | `n_special_tokens = 4` | `[PAD]`, `[EOS]`, etc.               |
        | `n_data_bins = 511`    | Main numeric bins                    |
        | `-1` adjustment        | Leaves room for one extra data token |

        """

        self.centers = torch.linspace(low_limit,
                                      high_limit,
                                      config.n_tokens - config.n_special_tokens - 1)
        
        # Boundary calculation:
        """
        This defines the edges of each bin - the boundaries between bins.

        Outline:

            1. Adjacent center points are averaged:
            
                boundary_i = (center_i + center_{i+1}) / 2 

            This gives boundaries between centers. 


            2. It prepends -infinity and appends +infinity
            so that every real number is in some bin.

            This creates a total of 512 boundaries with 511 bins:

                - Bins are defined as: 

                (-1e20, b_0), [b_0, b_1), ... [b_509, 1e20)

            Which means any real-valued input can be binned by findinng 
            which range it falls into. 

            * Takes real-valued time series inputs

            * Clips or scales them to lie in a defined range (low_limit to high_limit).

            * Uses uniform binning:
                - n evenly spaced bins defined by their centers.
                - Boundaries computed from adjacent center points. 
                - Each reall value will be mapped to the nearest bin center, 
                  which corresponds to a token ID.

        """
        # 1st argument of torch.concat() is a tuple of tensors:
        self.boundaries = torch.concat(
            (
            torch.tensor([-1e20], device = self.centers.device),
            (self.centers[1:] + self.centers [:-1]) /2, 
            torch.tensor([1e20], device = self.centers.device)
            )
        )

    
    def _input_transform(
            self, 
            context: torch.Tensor, 
            scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        """
        The method transforms raw time series input (real numbers) into
        token_ids - discretized integer tokens.

        Parameters
        ----------
        
        context
            A torch.Tensor of shape (batch_size, seq_len)
            (e.g. raw time series input, TODO: optionally add padding later with NaNs)

        scale
            Optional precomputed per-sample scale for normalization 
            (if not provided, the function computes it)

        """

        # ensure data is float32
        context = context.to(dtype = torch.float32)


        # Compute the scale if not provided:
        if scale is None:
            # torch.nansum() returns the sum of al elements treating NaNs as 0, 
            # of which we assume there aren't any since the data is clean.
            scale = torch.nansum(
                # computes the absolute value of each element:
                torch.abs(context), dim = -1 # sum along time axis
            ) / context.size(-1) # divide to get a mean absolute value per sample (roughly like L1 normalization)

            # set scale to 1.0 in cases where scale <= 0 to avoid division by 0.
            scale[~(scale>0)] = 1.0

            scale = scale.to(device=context.device)

            # now, scale shape is (batch_size, )
            # each item is the scale factor for one time series sample

        # normalize the context:
        scaled_context = context / scale.unsqueeze( dim = -1)
        # divide each value in the context by the corresponnding scale
        # unsqueezing from (batch_size, ) to (batch_size, 1) for broadcasting.


        """
        bucketize(
        input, 
        boundaries, ... right = False) -> Tensor:

        Returns the indices of the buckets to which each value in the input belongs,
        where the boundaries of the buckkets are set by ``boundaries``.


        Returns a new tensor with the same size as input.
        If ``right`` is False (default), then the left boundary is open.

        """

        token_ids = (torch.bucketize(input = scaled_context,
                                        boundaries= self.boundaries,
                                        right = True) + self.config.n_special_tokens)
        

        """
        This line clamps all values in the token_ids 
        tensor to be within the range [0, self.config.n_tokens - 1].
        It ensures that no token ID is less than 0 or 
        greater than the maximum allowed token index, which is self.config.n_tokens - 1.
        This is useful for preventing out-of-range token IDs after bucketization.
        """

        token_ids.clamp_(0, self.config.n_tokens -1)

        return token_ids

    def _append_eos_token(
            self,
            token_ids: torch.Tensor,
    ) -> torch.Tensor:
        
        batch_size = token_ids.shape[0]

        # Honestly not sure if I need this attention mask given that I know my data is clean and won't be padded...
        # attention_mask = ~torch.isnan(token_ids)
        # torch.full(size, fill_value, ...)
        # creates a tensor of size ``size`` filled with fill value.

        eos_tokens = torch.full((batch_size, 1),
                                fill_value= self.config.eos_token_id,
                                device = self.config.device)
        token_ids = torch.concat((token_ids, 
                                  eos_tokens), dim = 1)
        
        # eos_mask = torch.full((batch_size, 1), fill_value= True)

        # attention_mask = torch.concat((attention_mask, eos_mask), dim = 1)
        
        return token_ids #, attention_mask
    

    def context_input_transform(self, 
                                context: torch.Tensor) -> torch.Tensor:
        
        length = context.shape[-1]

        """
        The line context = context[..., -self.config.context_length:] 
        is a concise way to truncate the time series and keep only 
        the most recent context_length time steps along the last 
        dimension. 

        The ellipsis ... means "keep all preceding 
        dimensions unchanged," making the code flexible whether 
        the tensor shape is (B, T) or higher-dimensional like (B, F, T). 

        The slice 
        -self.config.context_length: 
        
        selects the last context_length values from the time axis,
        which is standard in time series modeling where recent data 
        is often the most informative. This ensures that, regardless 
        of the input length, the model always receives a 
        fixed-size window of the most recent inputs.
        """

        if length > self.config.context_length:
            context = context[..., -self. config.context_length :]

        token_ids = self._input_transform(context= context)
        
        if self.config.use_eos_token:
            token_ids = self._append_eos_token(token_ids= token_ids)

        return token_ids
    


class EncoderT5(nn.Module):
    """
    An ``EncoderT5`` wraps a ``PreTrainedModel`` object from ``transformers`` and uses 
    its final layer outputs for projecting to a final classification layer of dimension ``n_classes``. 

    Parameters
    ----------

    config
            The configuration to use.

    model 
            The pretrained model to use. (Really only T5 encoder only)

    """

    def __init__(self, config: ModelConfig,
                 model: PreTrainedModel) -> None:
        super().__init__()
        self.config = config 
        self.model = model

        hidden_size = self.model.config.hidden_size
        self.classification_layer = nn.Linear(hidden_size, config.n_classes)

    @property
    def device(self):
        return self.model.device 
    
    def encode(
            self, 
            input_ids: torch.Tensor,
    ):
        """
        Extract the encoder embeddings for the given token sequences. 

        Parameters
        ----------
        input_ids 
                Tensor of indices of input sequence tokens in the vocabulary 
                with shape (batch_size, seq_len). 

        
        Returns 
        -------
        embedding

                A tensor of encoder embeddings with shape 
                (batch_size, sequence_length, d_model).
        
        """

        assert (
            self.config.model_type == "seq2seq" or self.config.model_type == "encoderT5"
        ), "Encoder embeddings are only supported for encoder or encoder-decoder models."
        assert hasattr(self.model, "encoder")

        return self.model.encoder(
            input_ids = input_ids
        ).last_hidden_state
    
    def forward(self, 
                input_ids: torch.Tensor) -> torch.Tensor: 
        
        """
        Extract raw logits from the classification layer to use with Cross-entropy loss.


        Parameters
        ----------

        input_ids
                A tensor of token ids for a sequence. 

                
        Returns
        -------

        outputs
                A vector of raw logits that serve as input to the Cross-entropy loss.

        """

        # assert hasattr(self.model, "encode")

        embeddings = self.encode(input_ids = input_ids)

        outputs = self.classification_layer(embeddings)
        
        return outputs[:,-1]
    

 
