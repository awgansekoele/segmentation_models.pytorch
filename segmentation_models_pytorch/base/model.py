import torch
from . import initialization as init


class SegmentationModel(torch.nn.Module):
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def check_input_shape(self, x):

        l = x.shape[-1]
        output_stride = self.encoder.output_stride
        if l % output_stride != 0:
            new_l = (l // output_stride + 1) * output_stride if l % output_stride != 0 else l
            raise RuntimeError(
                f"Wrong input shape height={l}. Expected sequence length "
                f"divisible by {output_stride}. Consider padding your sequences to shape ({new_l})."
            )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: #D torch tensor with shape (batch_size, channels, length)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, length)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x
