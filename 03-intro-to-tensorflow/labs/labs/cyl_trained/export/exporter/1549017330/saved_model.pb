Ćť
ç
:
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
B
Equal
x"T
y"T
z
"
Ttype:
2	

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
D
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.8.02v1.8.0-0-g93bc2e20728Ż

global_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
k
global_step
VariableV2*
_class
loc:@global_step*
shape: *
dtype0	*
_output_shapes
: 

global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
T0	*
_class
loc:@global_step*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_output_shapes
: *
T0	*
_class
loc:@global_step
f
PlaceholderPlaceholder*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
h
Placeholder_1Placeholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

;dnn/input_from_feature_columns/input_layer/h/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
Á
7dnn/input_from_feature_columns/input_layer/h/ExpandDims
ExpandDimsPlaceholder;dnn/input_from_feature_columns/input_layer/h/ExpandDims/dim*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

2dnn/input_from_feature_columns/input_layer/h/ShapeShape7dnn/input_from_feature_columns/input_layer/h/ExpandDims*
T0*
_output_shapes
:

@dnn/input_from_feature_columns/input_layer/h/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Bdnn/input_from_feature_columns/input_layer/h/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:

Bdnn/input_from_feature_columns/input_layer/h/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

:dnn/input_from_feature_columns/input_layer/h/strided_sliceStridedSlice2dnn/input_from_feature_columns/input_layer/h/Shape@dnn/input_from_feature_columns/input_layer/h/strided_slice/stackBdnn/input_from_feature_columns/input_layer/h/strided_slice/stack_1Bdnn/input_from_feature_columns/input_layer/h/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: 
~
<dnn/input_from_feature_columns/input_layer/h/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
ę
:dnn/input_from_feature_columns/input_layer/h/Reshape/shapePack:dnn/input_from_feature_columns/input_layer/h/strided_slice<dnn/input_from_feature_columns/input_layer/h/Reshape/shape/1*
T0*
N*
_output_shapes
:
ć
4dnn/input_from_feature_columns/input_layer/h/ReshapeReshape7dnn/input_from_feature_columns/input_layer/h/ExpandDims:dnn/input_from_feature_columns/input_layer/h/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

;dnn/input_from_feature_columns/input_layer/r/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
Ă
7dnn/input_from_feature_columns/input_layer/r/ExpandDims
ExpandDimsPlaceholder_1;dnn/input_from_feature_columns/input_layer/r/ExpandDims/dim*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

2dnn/input_from_feature_columns/input_layer/r/ShapeShape7dnn/input_from_feature_columns/input_layer/r/ExpandDims*
T0*
_output_shapes
:

@dnn/input_from_feature_columns/input_layer/r/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 

Bdnn/input_from_feature_columns/input_layer/r/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Bdnn/input_from_feature_columns/input_layer/r/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

:dnn/input_from_feature_columns/input_layer/r/strided_sliceStridedSlice2dnn/input_from_feature_columns/input_layer/r/Shape@dnn/input_from_feature_columns/input_layer/r/strided_slice/stackBdnn/input_from_feature_columns/input_layer/r/strided_slice/stack_1Bdnn/input_from_feature_columns/input_layer/r/strided_slice/stack_2*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
~
<dnn/input_from_feature_columns/input_layer/r/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
ę
:dnn/input_from_feature_columns/input_layer/r/Reshape/shapePack:dnn/input_from_feature_columns/input_layer/r/strided_slice<dnn/input_from_feature_columns/input_layer/r/Reshape/shape/1*
T0*
N*
_output_shapes
:
ć
4dnn/input_from_feature_columns/input_layer/r/ReshapeReshape7dnn/input_from_feature_columns/input_layer/r/ExpandDims:dnn/input_from_feature_columns/input_layer/r/Reshape/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
x
6dnn/input_from_feature_columns/input_layer/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 

1dnn/input_from_feature_columns/input_layer/concatConcatV24dnn/input_from_feature_columns/input_layer/h/Reshape4dnn/input_from_feature_columns/input_layer/r/Reshape6dnn/input_from_feature_columns/input_layer/concat/axis*
T0*
N*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ĺ
@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"      *2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
:
ˇ
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *ý[ž*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
ˇ
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *ý[>*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0

Hdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
:	

>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: *
T0
­
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/sub*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
:	*
T0

:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
:	
Ľ
dnn/hiddenlayer_0/kernel/part_0
VariableV2*
dtype0*
_output_shapes
:	*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
shape:	
ë
&dnn/hiddenlayer_0/kernel/part_0/AssignAssigndnn/hiddenlayer_0/kernel/part_0:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform*
_output_shapes
:	*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0
Ż
$dnn/hiddenlayer_0/kernel/part_0/readIdentitydnn/hiddenlayer_0/kernel/part_0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
:	
°
/dnn/hiddenlayer_0/bias/part_0/Initializer/zerosConst*
valueB*    *0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes	
:

dnn/hiddenlayer_0/bias/part_0
VariableV2*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
shape:*
dtype0*
_output_shapes	
:
Ö
$dnn/hiddenlayer_0/bias/part_0/AssignAssigndnn/hiddenlayer_0/bias/part_0/dnn/hiddenlayer_0/bias/part_0/Initializer/zeros*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
_output_shapes	
:
Ľ
"dnn/hiddenlayer_0/bias/part_0/readIdentitydnn/hiddenlayer_0/bias/part_0*
_output_shapes	
:*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0
t
dnn/hiddenlayer_0/kernelIdentity$dnn/hiddenlayer_0/kernel/part_0/read*
_output_shapes
:	*
T0
˘
dnn/hiddenlayer_0/MatMulMatMul1dnn/input_from_feature_columns/input_layer/concatdnn/hiddenlayer_0/kernel*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
l
dnn/hiddenlayer_0/biasIdentity"dnn/hiddenlayer_0/bias/part_0/read*
_output_shapes	
:*
T0

dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/bias*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
dnn/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
dnn/zero_fraction/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 

dnn/zero_fraction/EqualEqualdnn/hiddenlayer_0/Reludnn/zero_fraction/zero*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
y
dnn/zero_fraction/CastCastdnn/zero_fraction/Equal*

SrcT0
*

DstT0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
dnn/zero_fraction/ConstConst*
_output_shapes
:*
valueB"       *
dtype0
p
dnn/zero_fraction/MeanMeandnn/zero_fraction/Castdnn/zero_fraction/Const*
T0*
_output_shapes
: 
 
2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_0/fraction_of_zero_values*
dtype0*
_output_shapes
: 
Ť
-dnn/dnn/hiddenlayer_0/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsdnn/zero_fraction/Mean*
T0*
_output_shapes
: 

$dnn/dnn/hiddenlayer_0/activation/tagConst*
dtype0*
_output_shapes
: *1
value(B& B dnn/dnn/hiddenlayer_0/activation

 dnn/dnn/hiddenlayer_0/activationHistogramSummary$dnn/dnn/hiddenlayer_0/activation/tagdnn/hiddenlayer_0/Relu*
_output_shapes
: 
Ĺ
@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"       *2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
:
ˇ
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *řKFž*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 
ˇ
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *řKF>*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0

Hdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shape*
_output_shapes
:	 *
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0

>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
_output_shapes
: *
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0
­
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/sub*
_output_shapes
:	 *
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0

:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
:	 
Ľ
dnn/hiddenlayer_1/kernel/part_0
VariableV2*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
shape:	 *
dtype0*
_output_shapes
:	 
ë
&dnn/hiddenlayer_1/kernel/part_0/AssignAssigndnn/hiddenlayer_1/kernel/part_0:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
:	 
Ż
$dnn/hiddenlayer_1/kernel/part_0/readIdentitydnn/hiddenlayer_1/kernel/part_0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
:	 
Ž
/dnn/hiddenlayer_1/bias/part_0/Initializer/zerosConst*
dtype0*
_output_shapes
: *
valueB *    *0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0

dnn/hiddenlayer_1/bias/part_0
VariableV2*
dtype0*
_output_shapes
: *0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
shape: 
Ő
$dnn/hiddenlayer_1/bias/part_0/AssignAssigndnn/hiddenlayer_1/bias/part_0/dnn/hiddenlayer_1/bias/part_0/Initializer/zeros*
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
_output_shapes
: 
¤
"dnn/hiddenlayer_1/bias/part_0/readIdentitydnn/hiddenlayer_1/bias/part_0*
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
_output_shapes
: 
t
dnn/hiddenlayer_1/kernelIdentity$dnn/hiddenlayer_1/kernel/part_0/read*
T0*
_output_shapes
:	 

dnn/hiddenlayer_1/MatMulMatMuldnn/hiddenlayer_0/Reludnn/hiddenlayer_1/kernel*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
k
dnn/hiddenlayer_1/biasIdentity"dnn/hiddenlayer_1/bias/part_0/read*
T0*
_output_shapes
: 

dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/bias*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
k
dnn/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
]
dnn/zero_fraction_1/zeroConst*
dtype0*
_output_shapes
: *
valueB
 *    

dnn/zero_fraction_1/EqualEqualdnn/hiddenlayer_1/Reludnn/zero_fraction_1/zero*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
|
dnn/zero_fraction_1/CastCastdnn/zero_fraction_1/Equal*

SrcT0
*

DstT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
j
dnn/zero_fraction_1/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
v
dnn/zero_fraction_1/MeanMeandnn/zero_fraction_1/Castdnn/zero_fraction_1/Const*
_output_shapes
: *
T0
 
2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_1/fraction_of_zero_values*
dtype0*
_output_shapes
: 
­
-dnn/dnn/hiddenlayer_1/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsdnn/zero_fraction_1/Mean*
T0*
_output_shapes
: 

$dnn/dnn/hiddenlayer_1/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_1/activation*
dtype0*
_output_shapes
: 

 dnn/dnn/hiddenlayer_1/activationHistogramSummary$dnn/dnn/hiddenlayer_1/activation/tagdnn/hiddenlayer_1/Relu*
_output_shapes
: 
Ĺ
@dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"       *2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
dtype0*
_output_shapes
:
ˇ
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *ěŃž*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
dtype0*
_output_shapes
: 
ˇ
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *ěŃ>*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
dtype0*
_output_shapes
: 

Hdnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
dtype0*
_output_shapes

: 

>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/min*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes
: *
T0
Ź
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/sub*
_output_shapes

: *
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0

:dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes

: 
Ł
dnn/hiddenlayer_2/kernel/part_0
VariableV2*
dtype0*
_output_shapes

: *2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
shape
: 
ę
&dnn/hiddenlayer_2/kernel/part_0/AssignAssigndnn/hiddenlayer_2/kernel/part_0:dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform*
_output_shapes

: *
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0
Ž
$dnn/hiddenlayer_2/kernel/part_0/readIdentitydnn/hiddenlayer_2/kernel/part_0*
_output_shapes

: *
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0
Ž
/dnn/hiddenlayer_2/bias/part_0/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0

dnn/hiddenlayer_2/bias/part_0
VariableV2*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
shape:*
dtype0*
_output_shapes
:
Ő
$dnn/hiddenlayer_2/bias/part_0/AssignAssigndnn/hiddenlayer_2/bias/part_0/dnn/hiddenlayer_2/bias/part_0/Initializer/zeros*
_output_shapes
:*
T0*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0
¤
"dnn/hiddenlayer_2/bias/part_0/readIdentitydnn/hiddenlayer_2/bias/part_0*
T0*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
_output_shapes
:
s
dnn/hiddenlayer_2/kernelIdentity$dnn/hiddenlayer_2/kernel/part_0/read*
_output_shapes

: *
T0

dnn/hiddenlayer_2/MatMulMatMuldnn/hiddenlayer_1/Reludnn/hiddenlayer_2/kernel*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
k
dnn/hiddenlayer_2/biasIdentity"dnn/hiddenlayer_2/bias/part_0/read*
T0*
_output_shapes
:

dnn/hiddenlayer_2/BiasAddBiasAdddnn/hiddenlayer_2/MatMuldnn/hiddenlayer_2/bias*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
k
dnn/hiddenlayer_2/ReluReludnn/hiddenlayer_2/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
]
dnn/zero_fraction_2/zeroConst*
_output_shapes
: *
valueB
 *    *
dtype0

dnn/zero_fraction_2/EqualEqualdnn/hiddenlayer_2/Reludnn/zero_fraction_2/zero*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
|
dnn/zero_fraction_2/CastCastdnn/zero_fraction_2/Equal*

SrcT0
*

DstT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
dnn/zero_fraction_2/ConstConst*
_output_shapes
:*
valueB"       *
dtype0
v
dnn/zero_fraction_2/MeanMeandnn/zero_fraction_2/Castdnn/zero_fraction_2/Const*
T0*
_output_shapes
: 
 
2dnn/dnn/hiddenlayer_2/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_2/fraction_of_zero_values*
dtype0*
_output_shapes
: 
­
-dnn/dnn/hiddenlayer_2/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_2/fraction_of_zero_values/tagsdnn/zero_fraction_2/Mean*
_output_shapes
: *
T0

$dnn/dnn/hiddenlayer_2/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_2/activation*
dtype0*
_output_shapes
: 

 dnn/dnn/hiddenlayer_2/activationHistogramSummary$dnn/dnn/hiddenlayer_2/activation/tagdnn/hiddenlayer_2/Relu*
_output_shapes
: 
ˇ
9dnn/logits/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"      *+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
:
Š
7dnn/logits/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *7ż*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
Š
7dnn/logits/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *7?*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
đ
Adnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform9dnn/logits/kernel/part_0/Initializer/random_uniform/shape*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes

:
ţ
7dnn/logits/kernel/part_0/Initializer/random_uniform/subSub7dnn/logits/kernel/part_0/Initializer/random_uniform/max7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes
: 

7dnn/logits/kernel/part_0/Initializer/random_uniform/mulMulAdnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniform7dnn/logits/kernel/part_0/Initializer/random_uniform/sub*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:*
T0

3dnn/logits/kernel/part_0/Initializer/random_uniformAdd7dnn/logits/kernel/part_0/Initializer/random_uniform/mul7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:

dnn/logits/kernel/part_0
VariableV2*+
_class!
loc:@dnn/logits/kernel/part_0*
shape
:*
dtype0*
_output_shapes

:
Î
dnn/logits/kernel/part_0/AssignAssigndnn/logits/kernel/part_03dnn/logits/kernel/part_0/Initializer/random_uniform*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:*
T0

dnn/logits/kernel/part_0/readIdentitydnn/logits/kernel/part_0*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:
 
(dnn/logits/bias/part_0/Initializer/zerosConst*
valueB*    *)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
:

dnn/logits/bias/part_0
VariableV2*
dtype0*
_output_shapes
:*)
_class
loc:@dnn/logits/bias/part_0*
shape:
š
dnn/logits/bias/part_0/AssignAssigndnn/logits/bias/part_0(dnn/logits/bias/part_0/Initializer/zeros*
T0*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
:

dnn/logits/bias/part_0/readIdentitydnn/logits/bias/part_0*
T0*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
:
e
dnn/logits/kernelIdentitydnn/logits/kernel/part_0/read*
T0*
_output_shapes

:
x
dnn/logits/MatMulMatMuldnn/hiddenlayer_2/Reludnn/logits/kernel*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
]
dnn/logits/biasIdentitydnn/logits/bias/part_0/read*
_output_shapes
:*
T0
s
dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/bias*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
]
dnn/zero_fraction_3/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 

dnn/zero_fraction_3/EqualEqualdnn/logits/BiasAdddnn/zero_fraction_3/zero*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
|
dnn/zero_fraction_3/CastCastdnn/zero_fraction_3/Equal*

SrcT0
*

DstT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
dnn/zero_fraction_3/ConstConst*
_output_shapes
:*
valueB"       *
dtype0
v
dnn/zero_fraction_3/MeanMeandnn/zero_fraction_3/Castdnn/zero_fraction_3/Const*
T0*
_output_shapes
: 

+dnn/dnn/logits/fraction_of_zero_values/tagsConst*7
value.B, B&dnn/dnn/logits/fraction_of_zero_values*
dtype0*
_output_shapes
: 

&dnn/dnn/logits/fraction_of_zero_valuesScalarSummary+dnn/dnn/logits/fraction_of_zero_values/tagsdnn/zero_fraction_3/Mean*
T0*
_output_shapes
: 
w
dnn/dnn/logits/activation/tagConst**
value!B Bdnn/dnn/logits/activation*
dtype0*
_output_shapes
: 
x
dnn/dnn/logits/activationHistogramSummarydnn/dnn/logits/activation/tagdnn/logits/BiasAdd*
_output_shapes
: 
W
dnn/head/logits/ShapeShapednn/logits/BiasAdd*
_output_shapes
:*
T0
k
)dnn/head/logits/assert_rank_at_least/rankConst*
value	B :*
dtype0*
_output_shapes
: 
[
Sdnn/head/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
L
Ddnn/head/logits/assert_rank_at_least/static_checks_determined_all_okNoOp
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_59d15e3722864e4f965abc3ed512cc3e/part*
dtype0*
_output_shapes
: 
d
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
¸
save/SaveV2/tensor_namesConst"/device:CPU:0*Ü
valueŇBĎ	Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/hiddenlayer_2/biasBdnn/hiddenlayer_2/kernelBdnn/logits/biasBdnn/logits/kernelBglobal_step*
dtype0*
_output_shapes
:	
Ö
save/SaveV2/shape_and_slicesConst"/device:CPU:0*w
valuenBl	B	128 0,128B2 128 0,2:0,128B32 0,32B128 32 0,128:0,32B4 0,4B32 4 0,32:0,4B1 0,1B4 1 0,4:0,1B *
dtype0*
_output_shapes
:	
˛
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices"dnn/hiddenlayer_0/bias/part_0/read$dnn/hiddenlayer_0/kernel/part_0/read"dnn/hiddenlayer_1/bias/part_0/read$dnn/hiddenlayer_1/kernel/part_0/read"dnn/hiddenlayer_2/bias/part_0/read$dnn/hiddenlayer_2/kernel/part_0/readdnn/logits/bias/part_0/readdnn/logits/kernel/part_0/readglobal_step"/device:CPU:0*
dtypes
2		
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: *
T0
 
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
_output_shapes
:*
T0
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
ť
save/RestoreV2/tensor_namesConst"/device:CPU:0*Ü
valueŇBĎ	Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/hiddenlayer_2/biasBdnn/hiddenlayer_2/kernelBdnn/logits/biasBdnn/logits/kernelBglobal_step*
dtype0*
_output_shapes
:	
Ů
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*w
valuenBl	B	128 0,128B2 128 0,2:0,128B32 0,32B128 32 0,128:0,32B4 0,4B32 4 0,32:0,4B1 0,1B4 1 0,4:0,1B *
dtype0*
_output_shapes
:	
ę
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2		*[
_output_shapesI
G::	: :	 :: :::

save/AssignAssigndnn/hiddenlayer_0/bias/part_0save/RestoreV2*
_output_shapes	
:*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0
¨
save/Assign_1Assigndnn/hiddenlayer_0/kernel/part_0save/RestoreV2:1*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
:	

save/Assign_2Assigndnn/hiddenlayer_1/bias/part_0save/RestoreV2:2*
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
_output_shapes
: 
¨
save/Assign_3Assigndnn/hiddenlayer_1/kernel/part_0save/RestoreV2:3*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
:	 

save/Assign_4Assigndnn/hiddenlayer_2/bias/part_0save/RestoreV2:4*
T0*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
_output_shapes
:
§
save/Assign_5Assigndnn/hiddenlayer_2/kernel/part_0save/RestoreV2:5*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes

: 

save/Assign_6Assigndnn/logits/bias/part_0save/RestoreV2:6*
T0*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
:

save/Assign_7Assigndnn/logits/kernel/part_0save/RestoreV2:7*
_output_shapes

:*
T0*+
_class!
loc:@dnn/logits/kernel/part_0
w
save/Assign_8Assignglobal_stepsave/RestoreV2:8*
T0	*
_class
loc:@global_step*
_output_shapes
: 
¨
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8
-
save/restore_allNoOp^save/restore_shard

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
R
save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_da11b336bfdb4d2bb4f245af8da8871a/part*
dtype0*
_output_shapes
: 
j
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
_output_shapes
: *
N
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
m
save_1/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 
ş
save_1/SaveV2/tensor_namesConst"/device:CPU:0*Ü
valueŇBĎ	Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/hiddenlayer_2/biasBdnn/hiddenlayer_2/kernelBdnn/logits/biasBdnn/logits/kernelBglobal_step*
dtype0*
_output_shapes
:	
Ř
save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*w
valuenBl	B	128 0,128B2 128 0,2:0,128B32 0,32B128 32 0,128:0,32B4 0,4B32 4 0,32:0,4B1 0,1B4 1 0,4:0,1B *
dtype0*
_output_shapes
:	
ş
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slices"dnn/hiddenlayer_0/bias/part_0/read$dnn/hiddenlayer_0/kernel/part_0/read"dnn/hiddenlayer_1/bias/part_0/read$dnn/hiddenlayer_1/kernel/part_0/read"dnn/hiddenlayer_2/bias/part_0/read$dnn/hiddenlayer_2/kernel/part_0/readdnn/logits/bias/part_0/readdnn/logits/kernel/part_0/readglobal_step"/device:CPU:0*
dtypes
2		
¨
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2"/device:CPU:0*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
Ś
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency"/device:CPU:0*
T0*
N*
_output_shapes
:
{
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const"/device:CPU:0

save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
˝
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*Ü
valueŇBĎ	Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/hiddenlayer_2/biasBdnn/hiddenlayer_2/kernelBdnn/logits/biasBdnn/logits/kernelBglobal_step*
dtype0*
_output_shapes
:	
Ű
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*w
valuenBl	B	128 0,128B2 128 0,2:0,128B32 0,32B128 32 0,128:0,32B4 0,4B32 4 0,32:0,4B1 0,1B4 1 0,4:0,1B *
dtype0*
_output_shapes
:	
ň
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*[
_output_shapesI
G::	: :	 :: :::*
dtypes
2		
 
save_1/AssignAssigndnn/hiddenlayer_0/bias/part_0save_1/RestoreV2*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
_output_shapes	
:*
T0
Ź
save_1/Assign_1Assigndnn/hiddenlayer_0/kernel/part_0save_1/RestoreV2:1*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
:	
Ł
save_1/Assign_2Assigndnn/hiddenlayer_1/bias/part_0save_1/RestoreV2:2*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
_output_shapes
: *
T0
Ź
save_1/Assign_3Assigndnn/hiddenlayer_1/kernel/part_0save_1/RestoreV2:3*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
:	 
Ł
save_1/Assign_4Assigndnn/hiddenlayer_2/bias/part_0save_1/RestoreV2:4*
T0*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
_output_shapes
:
Ť
save_1/Assign_5Assigndnn/hiddenlayer_2/kernel/part_0save_1/RestoreV2:5*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes

: 

save_1/Assign_6Assigndnn/logits/bias/part_0save_1/RestoreV2:6*
T0*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
:

save_1/Assign_7Assigndnn/logits/kernel/part_0save_1/RestoreV2:7*
_output_shapes

:*
T0*+
_class!
loc:@dnn/logits/kernel/part_0
{
save_1/Assign_8Assignglobal_stepsave_1/RestoreV2:8*
_class
loc:@global_step*
_output_shapes
: *
T0	
ź
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8
1
save_1/restore_allNoOp^save_1/restore_shard"B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"×
	summariesÉ
Ć
/dnn/dnn/hiddenlayer_0/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_0/activation:0
/dnn/dnn/hiddenlayer_1/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_1/activation:0
/dnn/dnn/hiddenlayer_2/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_2/activation:0
(dnn/dnn/logits/fraction_of_zero_values:0
dnn/dnn/logits/activation:0"ă
trainable_variablesËČ
Ű
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign&dnn/hiddenlayer_0/kernel/part_0/read:0"(
dnn/hiddenlayer_0/kernel  "2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:0
Ĺ
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign$dnn/hiddenlayer_0/bias/part_0/read:0"#
dnn/hiddenlayer_0/bias "21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:0
Ű
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign&dnn/hiddenlayer_1/kernel/part_0/read:0"(
dnn/hiddenlayer_1/kernel   " 2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:0
Ă
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign$dnn/hiddenlayer_1/bias/part_0/read:0"!
dnn/hiddenlayer_1/bias  " 21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:0
Ů
!dnn/hiddenlayer_2/kernel/part_0:0&dnn/hiddenlayer_2/kernel/part_0/Assign&dnn/hiddenlayer_2/kernel/part_0/read:0"&
dnn/hiddenlayer_2/kernel   " 2<dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform:0
Ă
dnn/hiddenlayer_2/bias/part_0:0$dnn/hiddenlayer_2/bias/part_0/Assign$dnn/hiddenlayer_2/bias/part_0/read:0"!
dnn/hiddenlayer_2/bias "21dnn/hiddenlayer_2/bias/part_0/Initializer/zeros:0
ś
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assigndnn/logits/kernel/part_0/read:0"
dnn/logits/kernel  "25dnn/logits/kernel/part_0/Initializer/random_uniform:0
 
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assigndnn/logits/bias/part_0/read:0"
dnn/logits/bias "2*dnn/logits/bias/part_0/Initializer/zeros:0"k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0"ł
	variablesĽ˘
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
Ű
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign&dnn/hiddenlayer_0/kernel/part_0/read:0"(
dnn/hiddenlayer_0/kernel  "2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:0
Ĺ
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign$dnn/hiddenlayer_0/bias/part_0/read:0"#
dnn/hiddenlayer_0/bias "21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:0
Ű
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign&dnn/hiddenlayer_1/kernel/part_0/read:0"(
dnn/hiddenlayer_1/kernel   " 2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:0
Ă
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign$dnn/hiddenlayer_1/bias/part_0/read:0"!
dnn/hiddenlayer_1/bias  " 21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:0
Ů
!dnn/hiddenlayer_2/kernel/part_0:0&dnn/hiddenlayer_2/kernel/part_0/Assign&dnn/hiddenlayer_2/kernel/part_0/read:0"&
dnn/hiddenlayer_2/kernel   " 2<dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform:0
Ă
dnn/hiddenlayer_2/bias/part_0:0$dnn/hiddenlayer_2/bias/part_0/Assign$dnn/hiddenlayer_2/bias/part_0/read:0"!
dnn/hiddenlayer_2/bias "21dnn/hiddenlayer_2/bias/part_0/Initializer/zeros:0
ś
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assigndnn/logits/kernel/part_0/read:0"
dnn/logits/kernel  "25dnn/logits/kernel/part_0/Initializer/random_uniform:0
 
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assigndnn/logits/bias/part_0/read:0"
dnn/logits/bias "2*dnn/logits/bias/part_0/Initializer/zeros:0" 
legacy_init_op


group_deps*´
predict¨
%
h 
Placeholder:0˙˙˙˙˙˙˙˙˙
'
r"
Placeholder_1:0˙˙˙˙˙˙˙˙˙:
predictions+
dnn/logits/BiasAdd:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict