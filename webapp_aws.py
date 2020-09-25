from __future__ import print_function
from __future__ import absolute_import

import os
import tensorflow as tf
import base64

tf.enable_eager_execution()
print(tf.executing_eagerly())

from Vocabulary import *
from Utils import *
from compiler.classes.Compiler import *

from seq2seqModel import *

# tf.test.is_gpu_available()

IMAGE_SIZE = 256
max_length= 37
attention_features_shape = 64
embedding_dim = 128
units = 512
vocab_size = 14
output_path = 'bin'
input_shape = (256, 256, 3)
voc = Vocabulary()
voc.retrieve(output_path)

def restore_sequence(token_sequence):
    first_row = True
#     a = token_sequence[1:]
    a = token_sequence
    b = []
    last_index = len(a) - 1
    b.append('<START>')
    b.append('stack')
    b.append('{')
    b.append('\n')
    for idx,token in enumerate(a):
           
        if first_row:
            first_row = False
            if token == 'row':
                b.append('row')
                b.append('{')
                b.append('\n')
            elif token == '<END>':
                b.append('}')
                b.append('\n')
                b.append('<END>')
            else:
                b.append('}')
                b.append('\n')
                b.append('footer')
                b.append('{')
                b.append('\n')

        else:
            if token == 'row':
                b.append('}')
                b.append('\n')
                b.append('row')
                b.append('{')
                b.append('\n')
            elif token == 'footer':
                b.append('}')
                b.append('\n')
                b.append('}')
                b.append('\n')
                b.append('footer')
                b.append('{')
                b.append('\n')
            elif token == '<END>':
                b.append('}')
                b.append('\n')
                b.append('<END>')
            else:
                b.append(token)
                if idx != last_index:
                    if a[idx+1] != 'row' and a[idx+1] != 'footer' and a[idx+1] != '<END>':
                        b.append(',')
                    else:
                        b.append('\n')
#         print(b)
    return b
 
def load_image(img):
    # pic = img * 255
    # pic = np.array(pic, dtype=np.uint8)
    # Utils.show(pic)
    img = Utils.get_preprocessed_img(img, IMAGE_SIZE)
    img = img.reshape([1] + list(input_shape))
    return img

def evaluate(image):
    # attention_plot = np.zeros((max_length, attention_features_shape))


    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)
    optimizer = tf.keras.optimizers.Adam(lr=0.0001, epsilon=1e-07) 

    model_name = 'vgg_in_loop'
    checkpoint_path = "{}/checkpoints/{}".format(output_path, model_name)
    ckpt = tf.train.Checkpoint(encoder=encoder,
                            decoder=decoder,
                            optimizer = optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
    else:
        print("error loading weights")

    hidden = decoder.reset_state(batch_size=1)

    features = encoder(image)

    dec_input = tf.expand_dims([0], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        # attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy() #escoger el máx

        result.append(voc.token_lookup[predicted_id])

        # print(voc.token_lookup[predicted_id])

        if voc.token_lookup[predicted_id] == '<END>':
            return result

        dec_input = tf.expand_dims([predicted_id], 0)

    # attention_plot = attention_plot[:len(result), :]
    return result


TEXT_PLACE_HOLDER = "[TEXT]"
ID_PLACE_HOLDER = "[ID]"

def render_content_with_text(key, value):
    value = value.replace(TEXT_PLACE_HOLDER, Utils.get_random_text(length_text=5, space_number=0))
    while value.find(ID_PLACE_HOLDER) != -1:
        value = value.replace(ID_PLACE_HOLDER, Utils.get_android_id(), 1)
    return value


def main():

    if len(sys.argv) < 2:
        print("Error")
        return

    eval_path = 'input';
    file_name = sys.argv[1]
    # file_name = 'deer_decode'

    # image_64_encode = image_64.split(',')[1]
    # image_64_decode = base64.b64decode(image_64_encode) 
    # image_result = open('deer_decode.png', 'wb') # create a writable image and write the decoding result
    # image_result.write(image_64_decode)

    image_path = "{}/{}.png".format(eval_path, file_name)
    # image_path = "{}".format(file_name)
    image = load_image(image_path)

    predictions = evaluate(image)
    token_sequence = restore_sequence(predictions)
    
    result = ''.join(token_sequence)
    print(result)
    # with open("{}/{}.gui".format('/mnt/datos/projects_hdd/web/react/pix2code/public/tmp/output', file_name), 'w') as out_f:
    #     out_f.write(result.replace('<START>', "").replace('<END>', ""))
    with open("{}/{}.gui".format('output', file_name), 'w') as out_f:
        out_f.write(result.replace('<START>', "").replace('<END>', ""))

    dsl_path = "compiler/assets/android-dsl-mapping.json"
    compiler = Compiler(dsl_path)

    input_file_path = "{}/{}.gui".format('output', file_name)
    output_file_path = "{}/{}.xml".format('output', file_name)

    compiler.compile(input_file_path, output_file_path, rendering_function=render_content_with_text)

    # if os.path.exists(image_path):
    #     # os.remove(image_path) # one file at a time
    #     os.remove(input_file_path) 

if __name__ == "__main__":
    main()