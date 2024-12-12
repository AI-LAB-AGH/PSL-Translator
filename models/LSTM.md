## LSTM structure described here

### Architecture

`self.l` - left hand <br />
`self.r` - right hand <br />
`self.f2l` - relation between left hand and face source <br />
`self.f2r` - relation between right hand and face source <br />
`self.h2h` - relation between left and right hand <br />
`self.fuse` - accepts the concatenated outputs of the 5 LSTMs and reduces the vector to length `hidden_size` <br />
`self.fc` - maps the feature vector of length `hidden_size` onto the vocabulary

### Relationships

`source_body` - center face keypoint ( COCO-WB <b>index 1</b> )<br />
`source_left` - first joint in left middle finger ( COCO-WhB <b>index 101</b> )<br />
`source_right` - first joint in right middle finger ( COCO-WB <b>index 122</b> )<br />
`body` - body keypoints less `source_body` <br />
`left` - left hand keypoints less `source_left` <br />
`right` - right hand keypoints less `source_right` <br />

### Normalization

`w_left`, `h_left` - normalization of `left` (dividing by hand width and height) <br />
`w_right`, `h_right` - normalization of `right` (dividing by hand width and height) <br />
`w_body`, `h_body` - normalization of `body` (dividing by hand width and height) <br />

### Improvement ideas for the network

- Architecture

  - Remove `self.fuse`
  - 1 LSTM for each finger
  - ???

- Normalization
  - Partially remove normalization
  - Remove normalization entirely

### Conducted experiments

|       MOD        | HIDDEN | CUT | SKIP | EPOCHS | RESULT                  | INFERENCE | CONCLUSION      |
| :--------------: | ------ | --- | ---- | ------ | ----------------------- | --------- | --------------- |
|   fuse removed   | 20     | 0   | 2    | 10     | training same as before | ?         | fuse not needed |
| depth introduced | 20     | 0   | 2    | 10     | ?                       | ?         | ?               |
