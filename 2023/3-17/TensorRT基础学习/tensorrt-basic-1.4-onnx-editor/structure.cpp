ir_version: 6
producer_name: "pytorch"
producer_version: "1.9"
graph {
  node {
    input: "image"
    input: "conv.weight"
    input: "conv.bias"
    output: "3" // 这里的 3 仅仅是个名字，但是中间过程的node的输入和输出一般仅用数字表示 ouput: "conv_output"但是不简洁
    name: "Conv_0"
    op_type: "Conv"
    attribute {
      name: "dilations"
      ints: 1
      ints: 1
      type: INTS
    }
    attribute {
      name: "group"
      i: 1
      type: INT
    }
    attribute {
      name: "kernel_shape"
      ints: 3
      ints: 3
      type: INTS
    }
    attribute {
      name: "pads"
      ints: 1
      ints: 1
      ints: 1
      ints: 1
      type: INTS
    }
    attribute {
      name: "strides"
      ints: 1
      ints: 1
      type: INTS
    }
  }
  node {
    input: "3"
    output: "output"
    name: "Relu_1"
    op_type: "Relu"
  }

  name: "torch-jit-export"
  initializer {
    dims: 1
    dims: 1
    dims: 3
    dims: 3
    data_type: 1
    name: "conv.weight"
    raw_data: "\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?"
  }
  initializer {
    dims: 1
    data_type: 1
    name: "conv.bias"
    raw_data: "\000\000\000\000"
  }
  input { // input 的 type
    name: "image"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_param: "batch"
          }
          dim {
            dim_value: 1
          }
          dim {
            dim_param: "height"
          }
          dim {
            dim_param: "width"
          }
        }
      }
    }
  }
  output { // output 的type
    name: "output"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_param: "batch"
          }
          dim {
            dim_value: 1
          }
          dim {
            dim_param: "height"
          }
          dim {
            dim_param: "width"
          }
        }
      }
    }
  }
}
opset_import {
  version: 11
}