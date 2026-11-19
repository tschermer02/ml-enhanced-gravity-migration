"""Independent copy of signed E11 3D U-Net architecture, same layer names."""
import tensorflow as tf

def _conv(x, filters, name):
    x=tf.keras.layers.Conv3D(filters,3,padding='same',activation='relu',name=f'{name}_conv1')(x)
    return tf.keras.layers.Conv3D(filters,3,padding='same',activation='relu',name=f'{name}_conv2')(x)

def build_model():
    x=tf.keras.Input((24,64,64,1),name='migration_input')
    e1=_conv(x,16,'encoder1')
    e2=_conv(tf.keras.layers.MaxPooling3D(2,name='pool1')(e1),32,'encoder2')
    b=_conv(tf.keras.layers.MaxPooling3D(2,name='pool2')(e2),64,'bottleneck')
    u2=_conv(tf.keras.layers.Concatenate(name='skip2')([tf.keras.layers.UpSampling3D(2,name='upsample2')(b),e2]),32,'decoder2')
    u1=_conv(tf.keras.layers.Concatenate(name='skip1')([tf.keras.layers.UpSampling3D(2,name='upsample1')(u2),e1]),16,'decoder1')
    y=tf.keras.layers.Conv3D(1,1,activation='tanh',name='signed_density_prediction')(u1)
    return tf.keras.Model(x,y,name='e12_signed_unet3d')
