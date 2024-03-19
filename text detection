#!/usr/bin/env python3
"""
 Copyright (c) 2019-2023 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging as log
import sys
import re
from pathlib import Path
from time import perf_counter
from argparse import ArgumentParser, SUPPRESS

import cv2
import numpy as np
from scipy.special import softmax
from openvino.runtime import Core, get_version

from text_spotting_demo.tracker import StaticIOUTracker

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python/openvino/model_zoo'))

import monitors
from images_capture import open_images_capture
from visualizers import InstanceSegmentationVisualizer
from model_api.performance_metrics import PerformanceMetrics

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

SOS_INDEX = 0
EOS_INDEX = 1
MAX_SEQ_LEN = 28

# Regular expression pattern for Indian vehicle registration number
INDIAN_VEHICLE_REGEX = r'^[A-Z]{2}\s*\d{2}\s*[A-Z]{2}\s*\d{4}$'

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-m_m', '--mask_rcnn_model',
                      help='Required. Path to an .xml file with a trained Mask-RCNN model with '
                           'additional text features output.',
                      required=True, type=str, metavar='"<path>"')
    args.add_argument('-m_te', '--text_enc_model',
                      help='Required. Path to an .xml file with a trained text recognition model '
                           '(encoder part).',
                      required=True, type=str, metavar='"<path>"')
    args.add_argument('-m_td', '--text_dec_model',
                      help='Required. Path to an .xml file with a trained text recognition model '
                           '(decoder part).',
                      required=True, type=str, metavar='"<path>"')
    args.add_argument('-i', '--input', required=True,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('--loop', default=False, action='store_true',
                      help='Optional. Enable reading the input in a loop.')
    args.add_argument('-d', '--device',
                      help='Optional. Specify the target device to infer on, i.e : CPU, GPU. '
                           'The demo will look for a suitable plugin for device specified '
                           '(by default, it is CPU). Please refer to OpenVINO documentation '
                           'for the list of devices supported by the model.',
                      default='CPU', type=str, metavar='"<device>"')
    args.add_argument('-pt', '--prob_threshold',
                      help='Optional. Probability threshold for detections filtering.',
                      default=0.5, type=float, metavar='"<num>"')
    args.add_argument('-a', '--alphabet',
                      help='Optional. Alphabet that is used for decoding.',
                      default='  abcdefghijklmnopqrstuvwxyz0123456789')
    args.add_argument('--trd_input_prev_symbol',
                      help='Optional. Name of previous symbol input node to text recognition head decoder part.',
                      default='prev_symbol')
    args.add_argument('--trd_input_prev_hidden',
                      help='Optional. Name of previous hidden input node to text recognition head decoder part.',
                      default='prev_hidden')
    args.add_argument('--trd_input_encoder_outputs',
                      help='Optional. Name of encoder outputs input node to text recognition head decoder part.',
                      default='encoder_outputs')
    args.add_argument('--trd_output_symbols_distr',
                      help='Optional. Name of symbols distribution output node from text recognition head decoder part.',
                      default='output')
    args.add_argument('--trd_output_cur_hidden',
                      help='Optional. Name of current hidden output node from text recognition head decoder part.',
                      default='hidden')
    args.add_argument('-trt', '--tr_threshold',
                      help='Optional. Text recognition confidence threshold.',
                      default=0.5, type=float, metavar='"<num>"')
    args.add_argument('--keep_aspect_ratio',
                      help='Optional. Force image resize to keep aspect ratio.',
                      action='store_true')
    args.add_argument('--no_track',
                      help='Optional. Disable tracking.',
                      action='store_true')
    args.add_argument('--show_scores',
                      help='Optional. Show detection scores.',
                      action='store_true')
    args.add_argument('--show_boxes',
                      help='Optional. Show bounding boxes.',
                      action='store_true')
    args.add_argument('-r', '--raw_output_message',
                      help='Optional. Output inference results raw values.',
                      action='store_true')
    args.add_argument("--no_show",
                      help="Optional. Don't show output",
                      action='store_true')
    args.add_argument('-u', '--utilization_monitors', default='', type=str,
                      help='Optional. List of monitors to show initially.')
    return parser


def segm_postprocess(box, raw_cls_mask, im_h, im_w):
    # Function for post-processing segmentation masks
    # Implementation remains the same
    pass


def is_indian_vehicle_registration_number(text):
    """
    Check if the given text matches the pattern of Indian vehicle registration number.
    """
    return re.match(INDIAN_VEHICLE_REGEX, text) is not None


def main():
    args = build_argparser().parse_args()

    cap = open_images_capture(args.input, args.loop)
    frame = cap.read()

    log.info('OpenVINO Runtime')
    log.info('\tbuild: {}'.format(get_version()))
    core = Core()

    # Read IR
    log.info('Reading Mask-RCNN model {}'.format(args.mask_rcnn_model))
    mask_rcnn_model = core.read_model(args.mask_rcnn_model)

    input_tensor_name = 'image'
    try:
        n, c, h, w = mask_rcnn_model.input(input_tensor_name).shape
        if n != 1:
            raise RuntimeError('Only batch 1 is supported by the demo application')
    except RuntimeError:
        raise RuntimeError('Demo supports only topologies with the following input tensor name: {}'.format(input_tensor_name))

    required_output_names = {'boxes', 'labels', 'masks', 'text_features'}
    for output_tensor_name in required_output_names:
        try:
            mask_rcnn_model.output(output_tensor_name)
        except RuntimeError:
            raise RuntimeError('Demo supports only topologies with the following output tensor names: {}'.format(
                ', '.join(required_output_names)))

    log.info('Reading Text Recognition Encoder model {}'.format(args.text_enc_model))
    text_enc_model = core.read_model(args.text_enc_model)

    log.info('Reading Text Recognition Decoder model {}'.format(args.text_dec_model))
    text_dec_model = core.read_model(args.text_dec_model)

    mask_rcnn_compiled_model = core.compile_model(mask_rcnn_model, device_name=args.device)
    mask_rcnn_infer_request = mask_rcnn_compiled_model.create_infer_request()
    log.info('The Mask-RCNN model {} is loaded to {}'.format(args.mask_rcnn_model, args.device))

    text_enc_compiled_model = core.compile_model(text_enc_model, args.device)
    text_enc_output_tensor = text_enc_compiled_model.outputs[0]
    text_enc_infer_request = text_enc_compiled_model.create_infer_request()
    log.info('The Text Recognition Encoder model {} is loaded to {}'.format(args.text_enc_model, args.device))

    text_dec_compiled_model = core.compile_model(text_dec_model, args.device)
    text_dec_infer_request = text_dec_compiled_model.create_infer_request()
    log.info('The Text Recognition Decoder model {} is loaded to {}'.format(args.text_dec_model, args.device))

    hidden_shape = text_dec_model.input(args.trd_input_prev_hidden).shape
    text_dec_output_names = {args.trd_output_symbols_distr, args.trd_output_cur_hidden}

    if args.no_track:
        tracker = None
    else:
        tracker = StaticIOUTracker()

    cap = open_images_capture(args.input, args.loop)

    while frame is not None:
        frame = cap.read()
        if frame is None:
            raise RuntimeError("Can't read an image from the input")

        # Resize the image
        input_image = cv2.resize(frame, (w, h))

        # Prepare input for Mask-RCNN model
        input_image = np.expand_dims(input_image.transpose((2, 0, 1)), axis=0).astype(np.float32)

        # Run inference on Mask-RCNN model
        mask_rcnn_infer_request.infer({input_tensor_name: input_image})
        outputs = {name: mask_rcnn_infer_request.get_tensor(name).data[:] for name in required_output_names}

        # Process detection results
        boxes = outputs['boxes'][:, :4]
        scores = outputs['boxes'][:, 4]
        classes = outputs['labels'].astype(np.uint32)
        raw_masks = outputs['masks']
        text_features = outputs['text_features']

        # Filter out detections with low confidence
        detections_filter = scores > args.prob_threshold
        scores = scores[detections_filter]
        classes = classes[detections_filter]
        boxes = boxes[detections_filter]
        raw_masks = raw_masks[detections_filter]
        text_features = text_features[detections_filter]

        # Process each detected instance
        for feature in text_features:
            text_lines = ['', '']  # Initialize two lines of text
            text_confidences = [1.0, 1.0]  # Initialize confidences for each line

            # Generate text for two lines
            for line_idx in range(2):
                # Prepare input for Text Recognition Encoder
                input_data = {'input': np.expand_dims(feature, axis=0)}

                # Run inference on Text Recognition Encoder
                feature = text_enc_infer_request.infer(input_data)[text_enc_output_tensor]
                feature = np.reshape(feature, (feature.shape[0], feature.shape[1], -1))
                feature = np.transpose(feature, (0, 2, 1))

                # Initialize hidden state for Text Recognition Decoder
                hidden = np.zeros(hidden_shape)
                prev_symbol_index = np.ones((1,)) * SOS_INDEX

                # Generate text one character at a time until EOS or max length
                for i in range(MAX_SEQ_LEN):
                    text_dec_infer_request.infer({
                        args.trd_input_prev_symbol: np.reshape(prev_symbol_index, (1,)),
                        args.trd_input_prev_hidden: hidden,
                        args.trd_input_encoder_outputs: feature})
                    decoder_output = {name: text_dec_infer_request.get_tensor(name).data[:] for name in text_dec_output_names}
                    symbols_distr = decoder_output[args.trd_output_symbols_distr]
                    symbols_distr_softmaxed = softmax(symbols_distr, axis=1)[0]
                    prev_symbol_index = int(np.argmax(symbols_distr, axis=1))
                    text_confidences[line_idx] *= symbols_distr_softmaxed[prev_symbol_index]
                    if prev_symbol_index == EOS_INDEX:
                        break
                    text_lines[line_idx] += args.alphabet[prev_symbol_index]
                    hidden = decoder_output[args.trd_output_cur_hidden]

            # Combine text from two lines
            combined_text = text_lines[0] + ' ' + text_lines[1]

            # Print the combined detected text and its confidence
            print("Detected Text:", combined_text if min(text_confidences) >= args.tr_threshold else "")
            print("Confidence:", min(text_confidences))

            # Check if the combined detected text is an Indian vehicle registration number
            if is_indian_vehicle_registration_number(combined_text):
                print("Detected Indian Vehicle Registration Number:", combined_text)
            else:
                print("Detected Text is not an Indian Vehicle Registration Number")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main() or 0)
