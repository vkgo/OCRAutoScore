import segmentation.Layout4Card.api as OuterSegmentation
import segmentation.blankSegmentation.blank_segmentation as BlankSegmentation
import scoreblocks.singleCharacterRecognition as SingleCharacterRecognition
import scoreblocks.fillblankmodel as FillBlankModel
import scoreblocks.candemo as CanDemo
class scoresystem:
    def __init__(self):
        self.outer_segmentation = OuterSegmentation.OuterSegmentation()
        self.blank_segmentation = BlankSegmentation.BlankSegmentation()
        self.single_character_recognition = SingleCharacterRecognition.Model('./scoreblocks/CharacetrRecognition/SpinalVGG.pth', 'SpinalVGG')
        self.fill_blank_model = FillBlankModel.model()




