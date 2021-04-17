import data_converter
import vgg19bn

dc = data_converter.DataConverter(vgg19bn.VGG19bn(layers=[60]))

dc.create_split(path='data', save_path='data2/in10_split')

dc.convert_split(path='data2/in10_split.npz', 
                 save_path='data2/in10_split_converted')