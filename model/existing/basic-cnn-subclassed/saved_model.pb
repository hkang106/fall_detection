ё╙
Щ¤
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.1.02unknown8ю╝

f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
Р
conv_net/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameconv_net/conv2d/kernel
Й
*conv_net/conv2d/kernel/Read/ReadVariableOpReadVariableOpconv_net/conv2d/kernel*&
_output_shapes
:*
dtype0
А
conv_net/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameconv_net/conv2d/bias
y
(conv_net/conv2d/bias/Read/ReadVariableOpReadVariableOpconv_net/conv2d/bias*
_output_shapes
:*
dtype0
Ф
conv_net/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv_net/conv2d_1/kernel
Н
,conv_net/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv_net/conv2d_1/kernel*&
_output_shapes
:*
dtype0
Д
conv_net/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameconv_net/conv2d_1/bias
}
*conv_net/conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv_net/conv2d_1/bias*
_output_shapes
:*
dtype0
Ф
conv_net/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameconv_net/conv2d_2/kernel
Н
,conv_net/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv_net/conv2d_2/kernel*&
_output_shapes
: *
dtype0
Д
conv_net/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameconv_net/conv2d_2/bias
}
*conv_net/conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv_net/conv2d_2/bias*
_output_shapes
: *
dtype0
Ф
conv_net/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *)
shared_nameconv_net/conv2d_3/kernel
Н
,conv_net/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv_net/conv2d_3/kernel*&
_output_shapes
:  *
dtype0
Д
conv_net/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameconv_net/conv2d_3/bias
}
*conv_net/conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv_net/conv2d_3/bias*
_output_shapes
: *
dtype0
Ф
conv_net/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*)
shared_nameconv_net/conv2d_4/kernel
Н
,conv_net/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv_net/conv2d_4/kernel*&
_output_shapes
: @*
dtype0
Д
conv_net/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameconv_net/conv2d_4/bias
}
*conv_net/conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv_net/conv2d_4/bias*
_output_shapes
:@*
dtype0
Ф
conv_net/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*)
shared_nameconv_net/conv2d_5/kernel
Н
,conv_net/conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv_net/conv2d_5/kernel*&
_output_shapes
:@@*
dtype0
Д
conv_net/conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameconv_net/conv2d_5/bias
}
*conv_net/conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv_net/conv2d_5/bias*
_output_shapes
:@*
dtype0
И
conv_net/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АHА*&
shared_nameconv_net/dense/kernel
Б
)conv_net/dense/kernel/Read/ReadVariableOpReadVariableOpconv_net/dense/kernel* 
_output_shapes
:
АHА*
dtype0

conv_net/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameconv_net/dense/bias
x
'conv_net/dense/bias/Read/ReadVariableOpReadVariableOpconv_net/dense/bias*
_output_shapes	
:А*
dtype0
Л
conv_net/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*(
shared_nameconv_net/dense_1/kernel
Д
+conv_net/dense_1/kernel/Read/ReadVariableOpReadVariableOpconv_net/dense_1/kernel*
_output_shapes
:	А*
dtype0
В
conv_net/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameconv_net/dense_1/bias
{
)conv_net/dense_1/bias/Read/ReadVariableOpReadVariableOpconv_net/dense_1/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
Ю
Adam/conv_net/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/conv_net/conv2d/kernel/m
Ч
1Adam/conv_net/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_net/conv2d/kernel/m*&
_output_shapes
:*
dtype0
О
Adam/conv_net/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/conv_net/conv2d/bias/m
З
/Adam/conv_net/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_net/conv2d/bias/m*
_output_shapes
:*
dtype0
в
Adam/conv_net/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv_net/conv2d_1/kernel/m
Ы
3Adam/conv_net/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_net/conv2d_1/kernel/m*&
_output_shapes
:*
dtype0
Т
Adam/conv_net/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/conv_net/conv2d_1/bias/m
Л
1Adam/conv_net/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_net/conv2d_1/bias/m*
_output_shapes
:*
dtype0
в
Adam/conv_net/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/conv_net/conv2d_2/kernel/m
Ы
3Adam/conv_net/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_net/conv2d_2/kernel/m*&
_output_shapes
: *
dtype0
Т
Adam/conv_net/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameAdam/conv_net/conv2d_2/bias/m
Л
1Adam/conv_net/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_net/conv2d_2/bias/m*
_output_shapes
: *
dtype0
в
Adam/conv_net/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *0
shared_name!Adam/conv_net/conv2d_3/kernel/m
Ы
3Adam/conv_net/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_net/conv2d_3/kernel/m*&
_output_shapes
:  *
dtype0
Т
Adam/conv_net/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameAdam/conv_net/conv2d_3/bias/m
Л
1Adam/conv_net/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_net/conv2d_3/bias/m*
_output_shapes
: *
dtype0
в
Adam/conv_net/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*0
shared_name!Adam/conv_net/conv2d_4/kernel/m
Ы
3Adam/conv_net/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_net/conv2d_4/kernel/m*&
_output_shapes
: @*
dtype0
Т
Adam/conv_net/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/conv_net/conv2d_4/bias/m
Л
1Adam/conv_net/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_net/conv2d_4/bias/m*
_output_shapes
:@*
dtype0
в
Adam/conv_net/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*0
shared_name!Adam/conv_net/conv2d_5/kernel/m
Ы
3Adam/conv_net/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_net/conv2d_5/kernel/m*&
_output_shapes
:@@*
dtype0
Т
Adam/conv_net/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/conv_net/conv2d_5/bias/m
Л
1Adam/conv_net/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_net/conv2d_5/bias/m*
_output_shapes
:@*
dtype0
Ц
Adam/conv_net/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АHА*-
shared_nameAdam/conv_net/dense/kernel/m
П
0Adam/conv_net/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_net/dense/kernel/m* 
_output_shapes
:
АHА*
dtype0
Н
Adam/conv_net/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_nameAdam/conv_net/dense/bias/m
Ж
.Adam/conv_net/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_net/dense/bias/m*
_output_shapes	
:А*
dtype0
Щ
Adam/conv_net/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*/
shared_name Adam/conv_net/dense_1/kernel/m
Т
2Adam/conv_net/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_net/dense_1/kernel/m*
_output_shapes
:	А*
dtype0
Р
Adam/conv_net/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/conv_net/dense_1/bias/m
Й
0Adam/conv_net/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_net/dense_1/bias/m*
_output_shapes
:*
dtype0
Ю
Adam/conv_net/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/conv_net/conv2d/kernel/v
Ч
1Adam/conv_net/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_net/conv2d/kernel/v*&
_output_shapes
:*
dtype0
О
Adam/conv_net/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/conv_net/conv2d/bias/v
З
/Adam/conv_net/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_net/conv2d/bias/v*
_output_shapes
:*
dtype0
в
Adam/conv_net/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv_net/conv2d_1/kernel/v
Ы
3Adam/conv_net/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_net/conv2d_1/kernel/v*&
_output_shapes
:*
dtype0
Т
Adam/conv_net/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/conv_net/conv2d_1/bias/v
Л
1Adam/conv_net/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_net/conv2d_1/bias/v*
_output_shapes
:*
dtype0
в
Adam/conv_net/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/conv_net/conv2d_2/kernel/v
Ы
3Adam/conv_net/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_net/conv2d_2/kernel/v*&
_output_shapes
: *
dtype0
Т
Adam/conv_net/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameAdam/conv_net/conv2d_2/bias/v
Л
1Adam/conv_net/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_net/conv2d_2/bias/v*
_output_shapes
: *
dtype0
в
Adam/conv_net/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *0
shared_name!Adam/conv_net/conv2d_3/kernel/v
Ы
3Adam/conv_net/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_net/conv2d_3/kernel/v*&
_output_shapes
:  *
dtype0
Т
Adam/conv_net/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameAdam/conv_net/conv2d_3/bias/v
Л
1Adam/conv_net/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_net/conv2d_3/bias/v*
_output_shapes
: *
dtype0
в
Adam/conv_net/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*0
shared_name!Adam/conv_net/conv2d_4/kernel/v
Ы
3Adam/conv_net/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_net/conv2d_4/kernel/v*&
_output_shapes
: @*
dtype0
Т
Adam/conv_net/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/conv_net/conv2d_4/bias/v
Л
1Adam/conv_net/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_net/conv2d_4/bias/v*
_output_shapes
:@*
dtype0
в
Adam/conv_net/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*0
shared_name!Adam/conv_net/conv2d_5/kernel/v
Ы
3Adam/conv_net/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_net/conv2d_5/kernel/v*&
_output_shapes
:@@*
dtype0
Т
Adam/conv_net/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/conv_net/conv2d_5/bias/v
Л
1Adam/conv_net/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_net/conv2d_5/bias/v*
_output_shapes
:@*
dtype0
Ц
Adam/conv_net/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АHА*-
shared_nameAdam/conv_net/dense/kernel/v
П
0Adam/conv_net/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_net/dense/kernel/v* 
_output_shapes
:
АHА*
dtype0
Н
Adam/conv_net/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_nameAdam/conv_net/dense/bias/v
Ж
.Adam/conv_net/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_net/dense/bias/v*
_output_shapes	
:А*
dtype0
Щ
Adam/conv_net/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*/
shared_name Adam/conv_net/dense_1/kernel/v
Т
2Adam/conv_net/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_net/dense_1/kernel/v*
_output_shapes
:	А*
dtype0
Р
Adam/conv_net/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/conv_net/dense_1/bias/v
Й
0Adam/conv_net/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_net/dense_1/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
╪U
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*УU
valueЙUBЖU B T

sequence
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
N
0
	1

2
3
4
5
6
7
8
9
10
А
iter

beta_1

beta_2
	decay
learning_ratemРmСmТmУmФmХmЦmЧ mШ!mЩ"mЪ#mЫ$mЬ%mЭ&mЮ'mЯvаvбvвvгvдvеvжvз vи!vй"vк#vл$vм%vн&vо'vп
v
0
1
2
3
4
5
6
7
 8
!9
"10
#11
$12
%13
&14
'15
v
0
1
2
3
4
5
6
7
 8
!9
"10
#11
$12
%13
&14
'15
 
Ъ
(metrics
trainable_variables
)layer_regularization_losses
	variables

*layers
+non_trainable_variables
regularization_losses
 
h

kernel
bias
,trainable_variables
-	variables
.regularization_losses
/	keras_api
h

kernel
bias
0trainable_variables
1	variables
2regularization_losses
3	keras_api
R
4trainable_variables
5	variables
6regularization_losses
7	keras_api
h

kernel
bias
8trainable_variables
9	variables
:regularization_losses
;	keras_api
h

kernel
bias
<trainable_variables
=	variables
>regularization_losses
?	keras_api
R
@trainable_variables
A	variables
Bregularization_losses
C	keras_api
h

 kernel
!bias
Dtrainable_variables
E	variables
Fregularization_losses
G	keras_api
h

"kernel
#bias
Htrainable_variables
I	variables
Jregularization_losses
K	keras_api
R
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
h

$kernel
%bias
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api
h

&kernel
'bias
Ttrainable_variables
U	variables
Vregularization_losses
W	keras_api
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEconv_net/conv2d/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv_net/conv2d/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv_net/conv2d_1/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEconv_net/conv2d_1/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv_net/conv2d_2/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEconv_net/conv2d_2/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv_net/conv2d_3/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEconv_net/conv2d_3/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv_net/conv2d_4/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEconv_net/conv2d_4/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv_net/conv2d_5/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEconv_net/conv2d_5/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEconv_net/dense/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv_net/dense/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv_net/dense_1/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEconv_net/dense_1/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE

X0
 
N
0
	1

2
3
4
5
6
7
8
9
10
 

0
1

0
1
 
Ъ
Ymetrics
,trainable_variables
Zlayer_regularization_losses
-	variables

[layers
\non_trainable_variables
.regularization_losses

0
1

0
1
 
Ъ
]metrics
0trainable_variables
^layer_regularization_losses
1	variables

_layers
`non_trainable_variables
2regularization_losses
 
 
 
Ъ
ametrics
4trainable_variables
blayer_regularization_losses
5	variables

clayers
dnon_trainable_variables
6regularization_losses

0
1

0
1
 
Ъ
emetrics
8trainable_variables
flayer_regularization_losses
9	variables

glayers
hnon_trainable_variables
:regularization_losses

0
1

0
1
 
Ъ
imetrics
<trainable_variables
jlayer_regularization_losses
=	variables

klayers
lnon_trainable_variables
>regularization_losses
 
 
 
Ъ
mmetrics
@trainable_variables
nlayer_regularization_losses
A	variables

olayers
pnon_trainable_variables
Bregularization_losses

 0
!1

 0
!1
 
Ъ
qmetrics
Dtrainable_variables
rlayer_regularization_losses
E	variables

slayers
tnon_trainable_variables
Fregularization_losses

"0
#1

"0
#1
 
Ъ
umetrics
Htrainable_variables
vlayer_regularization_losses
I	variables

wlayers
xnon_trainable_variables
Jregularization_losses
 
 
 
Ъ
ymetrics
Ltrainable_variables
zlayer_regularization_losses
M	variables

{layers
|non_trainable_variables
Nregularization_losses

$0
%1

$0
%1
 
Ы
}metrics
Ptrainable_variables
~layer_regularization_losses
Q	variables

layers
Аnon_trainable_variables
Rregularization_losses

&0
'1

&0
'1
 
Ю
Бmetrics
Ttrainable_variables
 Вlayer_regularization_losses
U	variables
Гlayers
Дnon_trainable_variables
Vregularization_losses


Еtotal

Жcount
З
_fn_kwargs
Иtrainable_variables
Й	variables
Кregularization_losses
Л	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

Е0
Ж1
 
б
Мmetrics
Иtrainable_variables
 Нlayer_regularization_losses
Й	variables
Оlayers
Пnon_trainable_variables
Кregularization_losses
 
 
 

Е0
Ж1
}
VARIABLE_VALUEAdam/conv_net/conv2d/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv_net/conv2d/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/conv_net/conv2d_1/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv_net/conv2d_1/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/conv_net/conv2d_2/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv_net/conv2d_2/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/conv_net/conv2d_3/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv_net/conv2d_3/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/conv_net/conv2d_4/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv_net/conv2d_4/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUEAdam/conv_net/conv2d_5/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEAdam/conv_net/conv2d_5/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv_net/dense/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv_net/dense/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/conv_net/dense_1/kernel/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv_net/dense_1/bias/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv_net/conv2d/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv_net/conv2d/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/conv_net/conv2d_1/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv_net/conv2d_1/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/conv_net/conv2d_2/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv_net/conv2d_2/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/conv_net/conv2d_3/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv_net/conv2d_3/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/conv_net/conv2d_4/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv_net/conv2d_4/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUEAdam/conv_net/conv2d_5/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEAdam/conv_net/conv2d_5/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv_net/dense/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv_net/dense/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/conv_net/dense_1/kernel/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv_net/dense_1/bias/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
К
serving_default_input_1Placeholder*/
_output_shapes
:         22*
dtype0*$
shape:         22
о
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv_net/conv2d/kernelconv_net/conv2d/biasconv_net/conv2d_1/kernelconv_net/conv2d_1/biasconv_net/conv2d_2/kernelconv_net/conv2d_2/biasconv_net/conv2d_3/kernelconv_net/conv2d_3/biasconv_net/conv2d_4/kernelconv_net/conv2d_4/biasconv_net/conv2d_5/kernelconv_net/conv2d_5/biasconv_net/dense/kernelconv_net/dense/biasconv_net/dense_1/kernelconv_net/dense_1/bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

GPU 

CPU2J 8*-
f(R&
$__inference_signature_wrapper_573962
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╟
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp*conv_net/conv2d/kernel/Read/ReadVariableOp(conv_net/conv2d/bias/Read/ReadVariableOp,conv_net/conv2d_1/kernel/Read/ReadVariableOp*conv_net/conv2d_1/bias/Read/ReadVariableOp,conv_net/conv2d_2/kernel/Read/ReadVariableOp*conv_net/conv2d_2/bias/Read/ReadVariableOp,conv_net/conv2d_3/kernel/Read/ReadVariableOp*conv_net/conv2d_3/bias/Read/ReadVariableOp,conv_net/conv2d_4/kernel/Read/ReadVariableOp*conv_net/conv2d_4/bias/Read/ReadVariableOp,conv_net/conv2d_5/kernel/Read/ReadVariableOp*conv_net/conv2d_5/bias/Read/ReadVariableOp)conv_net/dense/kernel/Read/ReadVariableOp'conv_net/dense/bias/Read/ReadVariableOp+conv_net/dense_1/kernel/Read/ReadVariableOp)conv_net/dense_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1Adam/conv_net/conv2d/kernel/m/Read/ReadVariableOp/Adam/conv_net/conv2d/bias/m/Read/ReadVariableOp3Adam/conv_net/conv2d_1/kernel/m/Read/ReadVariableOp1Adam/conv_net/conv2d_1/bias/m/Read/ReadVariableOp3Adam/conv_net/conv2d_2/kernel/m/Read/ReadVariableOp1Adam/conv_net/conv2d_2/bias/m/Read/ReadVariableOp3Adam/conv_net/conv2d_3/kernel/m/Read/ReadVariableOp1Adam/conv_net/conv2d_3/bias/m/Read/ReadVariableOp3Adam/conv_net/conv2d_4/kernel/m/Read/ReadVariableOp1Adam/conv_net/conv2d_4/bias/m/Read/ReadVariableOp3Adam/conv_net/conv2d_5/kernel/m/Read/ReadVariableOp1Adam/conv_net/conv2d_5/bias/m/Read/ReadVariableOp0Adam/conv_net/dense/kernel/m/Read/ReadVariableOp.Adam/conv_net/dense/bias/m/Read/ReadVariableOp2Adam/conv_net/dense_1/kernel/m/Read/ReadVariableOp0Adam/conv_net/dense_1/bias/m/Read/ReadVariableOp1Adam/conv_net/conv2d/kernel/v/Read/ReadVariableOp/Adam/conv_net/conv2d/bias/v/Read/ReadVariableOp3Adam/conv_net/conv2d_1/kernel/v/Read/ReadVariableOp1Adam/conv_net/conv2d_1/bias/v/Read/ReadVariableOp3Adam/conv_net/conv2d_2/kernel/v/Read/ReadVariableOp1Adam/conv_net/conv2d_2/bias/v/Read/ReadVariableOp3Adam/conv_net/conv2d_3/kernel/v/Read/ReadVariableOp1Adam/conv_net/conv2d_3/bias/v/Read/ReadVariableOp3Adam/conv_net/conv2d_4/kernel/v/Read/ReadVariableOp1Adam/conv_net/conv2d_4/bias/v/Read/ReadVariableOp3Adam/conv_net/conv2d_5/kernel/v/Read/ReadVariableOp1Adam/conv_net/conv2d_5/bias/v/Read/ReadVariableOp0Adam/conv_net/dense/kernel/v/Read/ReadVariableOp.Adam/conv_net/dense/bias/v/Read/ReadVariableOp2Adam/conv_net/dense_1/kernel/v/Read/ReadVariableOp0Adam/conv_net/dense_1/bias/v/Read/ReadVariableOpConst*D
Tin=
;29	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

GPU 

CPU2J 8*(
f#R!
__inference__traced_save_574368
Ў
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv_net/conv2d/kernelconv_net/conv2d/biasconv_net/conv2d_1/kernelconv_net/conv2d_1/biasconv_net/conv2d_2/kernelconv_net/conv2d_2/biasconv_net/conv2d_3/kernelconv_net/conv2d_3/biasconv_net/conv2d_4/kernelconv_net/conv2d_4/biasconv_net/conv2d_5/kernelconv_net/conv2d_5/biasconv_net/dense/kernelconv_net/dense/biasconv_net/dense_1/kernelconv_net/dense_1/biastotalcountAdam/conv_net/conv2d/kernel/mAdam/conv_net/conv2d/bias/mAdam/conv_net/conv2d_1/kernel/mAdam/conv_net/conv2d_1/bias/mAdam/conv_net/conv2d_2/kernel/mAdam/conv_net/conv2d_2/bias/mAdam/conv_net/conv2d_3/kernel/mAdam/conv_net/conv2d_3/bias/mAdam/conv_net/conv2d_4/kernel/mAdam/conv_net/conv2d_4/bias/mAdam/conv_net/conv2d_5/kernel/mAdam/conv_net/conv2d_5/bias/mAdam/conv_net/dense/kernel/mAdam/conv_net/dense/bias/mAdam/conv_net/dense_1/kernel/mAdam/conv_net/dense_1/bias/mAdam/conv_net/conv2d/kernel/vAdam/conv_net/conv2d/bias/vAdam/conv_net/conv2d_1/kernel/vAdam/conv_net/conv2d_1/bias/vAdam/conv_net/conv2d_2/kernel/vAdam/conv_net/conv2d_2/bias/vAdam/conv_net/conv2d_3/kernel/vAdam/conv_net/conv2d_3/bias/vAdam/conv_net/conv2d_4/kernel/vAdam/conv_net/conv2d_4/bias/vAdam/conv_net/conv2d_5/kernel/vAdam/conv_net/conv2d_5/bias/vAdam/conv_net/dense/kernel/vAdam/conv_net/dense/bias/vAdam/conv_net/dense_1/kernel/vAdam/conv_net/dense_1/bias/v*C
Tin<
:28*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

GPU 

CPU2J 8*+
f&R$
"__inference__traced_restore_574545в┬
└
к
)__inference_conv2d_4_layer_call_fn_573690

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_5736822
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
└
к
)__inference_conv2d_2_layer_call_fn_573636

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                            **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_5736282
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╩	
┌
A__inference_dense_layer_call_and_return_conditional_losses_574154

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АHА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         АH::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
ШR
╢

D__inference_conv_net_layer_call_and_return_conditional_losses_574026
x)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityИвconv2d/BiasAdd/ReadVariableOpвconv2d/Conv2D/ReadVariableOpвconv2d_1/BiasAdd/ReadVariableOpвconv2d_1/Conv2D/ReadVariableOpвconv2d_2/BiasAdd/ReadVariableOpвconv2d_2/Conv2D/ReadVariableOpвconv2d_3/BiasAdd/ReadVariableOpвconv2d_3/Conv2D/ReadVariableOpвconv2d_4/BiasAdd/ReadVariableOpвconv2d_4/Conv2D/ReadVariableOpвconv2d_5/BiasAdd/ReadVariableOpвconv2d_5/Conv2D/ReadVariableOpвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpк
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp│
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         22*
paddingSAME*
strides
2
conv2d/Conv2Dб
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOpд
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         222
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         222
conv2d/Relu░
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOp╤
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         22*
paddingSAME*
strides
2
conv2d_1/Conv2Dз
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpм
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         222
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         222
conv2d_1/Relu├
max_pooling2d/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool░
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_2/Conv2D/ReadVariableOp╓
conv2d_2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_2/Conv2Dз
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOpм
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:          2
conv2d_2/Relu░
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_3/Conv2D/ReadVariableOp╙
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_3/Conv2Dз
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_3/BiasAdd/ReadVariableOpм
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:          2
conv2d_3/Relu╟
max_pooling2d_1/MaxPoolMaxPoolconv2d_3/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool░
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_4/Conv2D/ReadVariableOp╪
conv2d_4/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_4/Conv2Dз
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_4/BiasAdd/ReadVariableOpм
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_4/BiasAdd{
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d_4/Relu░
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_5/Conv2D/ReadVariableOp╙
conv2d_5/Conv2DConv2Dconv2d_4/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_5/Conv2Dз
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_5/BiasAdd/ReadVariableOpм
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_5/BiasAdd{
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d_5/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"     $  2
flatten/ConstХ
flatten/ReshapeReshapeconv2d_5/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:         АH2
flatten/Reshapeб
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
АHА*
dtype02
dense/MatMul/ReadVariableOpШ
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense/MatMulЯ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
dense/BiasAdd/ReadVariableOpЪ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:         А2

dense/Reluж
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
dense_1/MatMul/ReadVariableOpЭ
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/MatMulд
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_1/Softmax∙
IdentityIdentitydense_1/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:         22::::::::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:! 

_user_specified_namex
ч6
║
D__inference_conv_net_layer_call_and_return_conditional_losses_573796
input_1)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2+
'conv2d_5_statefulpartitionedcall_args_1+
'conv2d_5_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallв conv2d_3/StatefulPartitionedCallв conv2d_4/StatefulPartitionedCallв conv2d_5/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallй
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         22**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_5735742 
conv2d/StatefulPartitionedCall╙
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         22**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_5735952"
 conv2d_1/StatefulPartitionedCall°
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_5736092
max_pooling2d/PartitionedCall╥
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:          **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_5736282"
 conv2d_2/StatefulPartitionedCall╒
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:          **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_5736492"
 conv2d_3/StatefulPartitionedCall■
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:          **
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_5736632!
max_pooling2d_1/PartitionedCall╘
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_5736822"
 conv2d_4/StatefulPartitionedCall╒
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0'conv2d_5_statefulpartitionedcall_args_1'conv2d_5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_5737032"
 conv2d_5/StatefulPartitionedCall▀
flatten/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         АH**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_5737412
flatten/PartitionedCall╢
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_5737602
dense/StatefulPartitionedCall┼
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_5737832!
dense_1/StatefulPartitionedCallО
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:         22::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
с
е
$__inference_signature_wrapper_573962
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16
identityИвStatefulPartitionedCall╣
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

GPU 

CPU2J 8**
f%R#
!__inference__wrapped_model_5735612
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:         22::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
└
к
)__inference_conv2d_3_layer_call_fn_573657

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                            **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_5736492
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ш
▌
D__inference_conv2d_4_layer_call_and_return_conditional_losses_573682

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp╡
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           @2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
└
к
)__inference_conv2d_5_layer_call_fn_573711

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_5737032
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           @::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
│
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_573609

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
К
_
C__inference_flatten_layer_call_and_return_conditional_losses_574138

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"     $  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         АH2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         АH2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:& "
 
_user_specified_nameinputs
ё
й
(__inference_dense_1_layer_call_fn_574179

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_5737832
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╡
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_573663

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
ш
▌
D__inference_conv2d_1_layer_call_and_return_conditional_losses_573595

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp╡
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
Й
к
)__inference_conv_net_layer_call_fn_573932
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16
identityИвStatefulPartitionedCall▄
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv_net_layer_call_and_return_conditional_losses_5739132
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:         22::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
╬	
▄
C__inference_dense_1_layer_call_and_return_conditional_losses_574172

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Й
к
)__inference_conv_net_layer_call_fn_573880
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16
identityИвStatefulPartitionedCall▄
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv_net_layer_call_and_return_conditional_losses_5738612
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:         22::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
└
к
)__inference_conv2d_1_layer_call_fn_573603

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_5735952
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ў
д
)__inference_conv_net_layer_call_fn_574132
x"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16
identityИвStatefulPartitionedCall╓
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv_net_layer_call_and_return_conditional_losses_5739132
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:         22::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
ШR
╢

D__inference_conv_net_layer_call_and_return_conditional_losses_574090
x)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityИвconv2d/BiasAdd/ReadVariableOpвconv2d/Conv2D/ReadVariableOpвconv2d_1/BiasAdd/ReadVariableOpвconv2d_1/Conv2D/ReadVariableOpвconv2d_2/BiasAdd/ReadVariableOpвconv2d_2/Conv2D/ReadVariableOpвconv2d_3/BiasAdd/ReadVariableOpвconv2d_3/Conv2D/ReadVariableOpвconv2d_4/BiasAdd/ReadVariableOpвconv2d_4/Conv2D/ReadVariableOpвconv2d_5/BiasAdd/ReadVariableOpвconv2d_5/Conv2D/ReadVariableOpвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpк
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp│
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         22*
paddingSAME*
strides
2
conv2d/Conv2Dб
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOpд
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         222
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         222
conv2d/Relu░
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOp╤
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         22*
paddingSAME*
strides
2
conv2d_1/Conv2Dз
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpм
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         222
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         222
conv2d_1/Relu├
max_pooling2d/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool░
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_2/Conv2D/ReadVariableOp╓
conv2d_2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_2/Conv2Dз
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOpм
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:          2
conv2d_2/Relu░
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_3/Conv2D/ReadVariableOp╙
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_3/Conv2Dз
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_3/BiasAdd/ReadVariableOpм
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:          2
conv2d_3/Relu╟
max_pooling2d_1/MaxPoolMaxPoolconv2d_3/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool░
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_4/Conv2D/ReadVariableOp╪
conv2d_4/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_4/Conv2Dз
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_4/BiasAdd/ReadVariableOpм
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_4/BiasAdd{
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d_4/Relu░
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_5/Conv2D/ReadVariableOp╙
conv2d_5/Conv2DConv2Dconv2d_4/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_5/Conv2Dз
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_5/BiasAdd/ReadVariableOpм
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_5/BiasAdd{
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d_5/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"     $  2
flatten/ConstХ
flatten/ReshapeReshapeconv2d_5/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:         АH2
flatten/Reshapeб
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
АHА*
dtype02
dense/MatMul/ReadVariableOpШ
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense/MatMulЯ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
dense/BiasAdd/ReadVariableOpЪ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:         А2

dense/Reluж
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
dense_1/MatMul/ReadVariableOpЭ
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/MatMulд
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_1/Softmax∙
IdentityIdentitydense_1/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:         22::::::::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:! 

_user_specified_namex
█
D
(__inference_flatten_layer_call_fn_574143

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         АH**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_5737412
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         АH2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:& "
 
_user_specified_nameinputs
╝
и
'__inference_conv2d_layer_call_fn_573582

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           **
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_5735742
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╒6
┤
D__inference_conv_net_layer_call_and_return_conditional_losses_573861
x)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2+
'conv2d_5_statefulpartitionedcall_args_1+
'conv2d_5_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallв conv2d_3/StatefulPartitionedCallв conv2d_4/StatefulPartitionedCallв conv2d_5/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallг
conv2d/StatefulPartitionedCallStatefulPartitionedCallx%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         22**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_5735742 
conv2d/StatefulPartitionedCall╙
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         22**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_5735952"
 conv2d_1/StatefulPartitionedCall°
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_5736092
max_pooling2d/PartitionedCall╥
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:          **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_5736282"
 conv2d_2/StatefulPartitionedCall╒
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:          **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_5736492"
 conv2d_3/StatefulPartitionedCall■
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:          **
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_5736632!
max_pooling2d_1/PartitionedCall╘
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_5736822"
 conv2d_4/StatefulPartitionedCall╒
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0'conv2d_5_statefulpartitionedcall_args_1'conv2d_5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_5737032"
 conv2d_5/StatefulPartitionedCall▀
flatten/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         АH**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_5737412
flatten/PartitionedCall╢
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_5737602
dense/StatefulPartitionedCall┼
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_5737832!
dense_1/StatefulPartitionedCallО
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:         22::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:! 

_user_specified_namex
ш
▌
D__inference_conv2d_5_layer_call_and_return_conditional_losses_573703

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp╡
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           @2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
я
з
&__inference_dense_layer_call_fn_574161

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_5737602
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         АH::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ц
█
B__inference_conv2d_layer_call_and_return_conditional_losses_573574

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp╡
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
Фb
╣
!__inference__wrapped_model_573561
input_12
.conv_net_conv2d_conv2d_readvariableop_resource3
/conv_net_conv2d_biasadd_readvariableop_resource4
0conv_net_conv2d_1_conv2d_readvariableop_resource5
1conv_net_conv2d_1_biasadd_readvariableop_resource4
0conv_net_conv2d_2_conv2d_readvariableop_resource5
1conv_net_conv2d_2_biasadd_readvariableop_resource4
0conv_net_conv2d_3_conv2d_readvariableop_resource5
1conv_net_conv2d_3_biasadd_readvariableop_resource4
0conv_net_conv2d_4_conv2d_readvariableop_resource5
1conv_net_conv2d_4_biasadd_readvariableop_resource4
0conv_net_conv2d_5_conv2d_readvariableop_resource5
1conv_net_conv2d_5_biasadd_readvariableop_resource1
-conv_net_dense_matmul_readvariableop_resource2
.conv_net_dense_biasadd_readvariableop_resource3
/conv_net_dense_1_matmul_readvariableop_resource4
0conv_net_dense_1_biasadd_readvariableop_resource
identityИв&conv_net/conv2d/BiasAdd/ReadVariableOpв%conv_net/conv2d/Conv2D/ReadVariableOpв(conv_net/conv2d_1/BiasAdd/ReadVariableOpв'conv_net/conv2d_1/Conv2D/ReadVariableOpв(conv_net/conv2d_2/BiasAdd/ReadVariableOpв'conv_net/conv2d_2/Conv2D/ReadVariableOpв(conv_net/conv2d_3/BiasAdd/ReadVariableOpв'conv_net/conv2d_3/Conv2D/ReadVariableOpв(conv_net/conv2d_4/BiasAdd/ReadVariableOpв'conv_net/conv2d_4/Conv2D/ReadVariableOpв(conv_net/conv2d_5/BiasAdd/ReadVariableOpв'conv_net/conv2d_5/Conv2D/ReadVariableOpв%conv_net/dense/BiasAdd/ReadVariableOpв$conv_net/dense/MatMul/ReadVariableOpв'conv_net/dense_1/BiasAdd/ReadVariableOpв&conv_net/dense_1/MatMul/ReadVariableOp┼
%conv_net/conv2d/Conv2D/ReadVariableOpReadVariableOp.conv_net_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv_net/conv2d/Conv2D/ReadVariableOp╘
conv_net/conv2d/Conv2DConv2Dinput_1-conv_net/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         22*
paddingSAME*
strides
2
conv_net/conv2d/Conv2D╝
&conv_net/conv2d/BiasAdd/ReadVariableOpReadVariableOp/conv_net_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv_net/conv2d/BiasAdd/ReadVariableOp╚
conv_net/conv2d/BiasAddBiasAddconv_net/conv2d/Conv2D:output:0.conv_net/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         222
conv_net/conv2d/BiasAddР
conv_net/conv2d/ReluRelu conv_net/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         222
conv_net/conv2d/Relu╦
'conv_net/conv2d_1/Conv2D/ReadVariableOpReadVariableOp0conv_net_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'conv_net/conv2d_1/Conv2D/ReadVariableOpї
conv_net/conv2d_1/Conv2DConv2D"conv_net/conv2d/Relu:activations:0/conv_net/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         22*
paddingSAME*
strides
2
conv_net/conv2d_1/Conv2D┬
(conv_net/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp1conv_net_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(conv_net/conv2d_1/BiasAdd/ReadVariableOp╨
conv_net/conv2d_1/BiasAddBiasAdd!conv_net/conv2d_1/Conv2D:output:00conv_net/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         222
conv_net/conv2d_1/BiasAddЦ
conv_net/conv2d_1/ReluRelu"conv_net/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         222
conv_net/conv2d_1/Relu▐
conv_net/max_pooling2d/MaxPoolMaxPool$conv_net/conv2d_1/Relu:activations:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
2 
conv_net/max_pooling2d/MaxPool╦
'conv_net/conv2d_2/Conv2D/ReadVariableOpReadVariableOp0conv_net_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'conv_net/conv2d_2/Conv2D/ReadVariableOp·
conv_net/conv2d_2/Conv2DConv2D'conv_net/max_pooling2d/MaxPool:output:0/conv_net/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv_net/conv2d_2/Conv2D┬
(conv_net/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp1conv_net_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(conv_net/conv2d_2/BiasAdd/ReadVariableOp╨
conv_net/conv2d_2/BiasAddBiasAdd!conv_net/conv2d_2/Conv2D:output:00conv_net/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv_net/conv2d_2/BiasAddЦ
conv_net/conv2d_2/ReluRelu"conv_net/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:          2
conv_net/conv2d_2/Relu╦
'conv_net/conv2d_3/Conv2D/ReadVariableOpReadVariableOp0conv_net_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02)
'conv_net/conv2d_3/Conv2D/ReadVariableOpў
conv_net/conv2d_3/Conv2DConv2D$conv_net/conv2d_2/Relu:activations:0/conv_net/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv_net/conv2d_3/Conv2D┬
(conv_net/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp1conv_net_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(conv_net/conv2d_3/BiasAdd/ReadVariableOp╨
conv_net/conv2d_3/BiasAddBiasAdd!conv_net/conv2d_3/Conv2D:output:00conv_net/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv_net/conv2d_3/BiasAddЦ
conv_net/conv2d_3/ReluRelu"conv_net/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:          2
conv_net/conv2d_3/Reluт
 conv_net/max_pooling2d_1/MaxPoolMaxPool$conv_net/conv2d_3/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2"
 conv_net/max_pooling2d_1/MaxPool╦
'conv_net/conv2d_4/Conv2D/ReadVariableOpReadVariableOp0conv_net_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02)
'conv_net/conv2d_4/Conv2D/ReadVariableOp№
conv_net/conv2d_4/Conv2DConv2D)conv_net/max_pooling2d_1/MaxPool:output:0/conv_net/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv_net/conv2d_4/Conv2D┬
(conv_net/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp1conv_net_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(conv_net/conv2d_4/BiasAdd/ReadVariableOp╨
conv_net/conv2d_4/BiasAddBiasAdd!conv_net/conv2d_4/Conv2D:output:00conv_net/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv_net/conv2d_4/BiasAddЦ
conv_net/conv2d_4/ReluRelu"conv_net/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv_net/conv2d_4/Relu╦
'conv_net/conv2d_5/Conv2D/ReadVariableOpReadVariableOp0conv_net_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'conv_net/conv2d_5/Conv2D/ReadVariableOpў
conv_net/conv2d_5/Conv2DConv2D$conv_net/conv2d_4/Relu:activations:0/conv_net/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv_net/conv2d_5/Conv2D┬
(conv_net/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp1conv_net_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(conv_net/conv2d_5/BiasAdd/ReadVariableOp╨
conv_net/conv2d_5/BiasAddBiasAdd!conv_net/conv2d_5/Conv2D:output:00conv_net/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv_net/conv2d_5/BiasAddЦ
conv_net/conv2d_5/ReluRelu"conv_net/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv_net/conv2d_5/ReluБ
conv_net/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"     $  2
conv_net/flatten/Const╣
conv_net/flatten/ReshapeReshape$conv_net/conv2d_5/Relu:activations:0conv_net/flatten/Const:output:0*
T0*(
_output_shapes
:         АH2
conv_net/flatten/Reshape╝
$conv_net/dense/MatMul/ReadVariableOpReadVariableOp-conv_net_dense_matmul_readvariableop_resource* 
_output_shapes
:
АHА*
dtype02&
$conv_net/dense/MatMul/ReadVariableOp╝
conv_net/dense/MatMulMatMul!conv_net/flatten/Reshape:output:0,conv_net/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
conv_net/dense/MatMul║
%conv_net/dense/BiasAdd/ReadVariableOpReadVariableOp.conv_net_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02'
%conv_net/dense/BiasAdd/ReadVariableOp╛
conv_net/dense/BiasAddBiasAddconv_net/dense/MatMul:product:0-conv_net/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
conv_net/dense/BiasAddЖ
conv_net/dense/ReluReluconv_net/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
conv_net/dense/Relu┴
&conv_net/dense_1/MatMul/ReadVariableOpReadVariableOp/conv_net_dense_1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02(
&conv_net/dense_1/MatMul/ReadVariableOp┴
conv_net/dense_1/MatMulMatMul!conv_net/dense/Relu:activations:0.conv_net/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
conv_net/dense_1/MatMul┐
'conv_net/dense_1/BiasAdd/ReadVariableOpReadVariableOp0conv_net_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'conv_net/dense_1/BiasAdd/ReadVariableOp┼
conv_net/dense_1/BiasAddBiasAdd!conv_net/dense_1/MatMul:product:0/conv_net/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
conv_net/dense_1/BiasAddФ
conv_net/dense_1/SoftmaxSoftmax!conv_net/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         2
conv_net/dense_1/SoftmaxТ
IdentityIdentity"conv_net/dense_1/Softmax:softmax:0'^conv_net/conv2d/BiasAdd/ReadVariableOp&^conv_net/conv2d/Conv2D/ReadVariableOp)^conv_net/conv2d_1/BiasAdd/ReadVariableOp(^conv_net/conv2d_1/Conv2D/ReadVariableOp)^conv_net/conv2d_2/BiasAdd/ReadVariableOp(^conv_net/conv2d_2/Conv2D/ReadVariableOp)^conv_net/conv2d_3/BiasAdd/ReadVariableOp(^conv_net/conv2d_3/Conv2D/ReadVariableOp)^conv_net/conv2d_4/BiasAdd/ReadVariableOp(^conv_net/conv2d_4/Conv2D/ReadVariableOp)^conv_net/conv2d_5/BiasAdd/ReadVariableOp(^conv_net/conv2d_5/Conv2D/ReadVariableOp&^conv_net/dense/BiasAdd/ReadVariableOp%^conv_net/dense/MatMul/ReadVariableOp(^conv_net/dense_1/BiasAdd/ReadVariableOp'^conv_net/dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:         22::::::::::::::::2P
&conv_net/conv2d/BiasAdd/ReadVariableOp&conv_net/conv2d/BiasAdd/ReadVariableOp2N
%conv_net/conv2d/Conv2D/ReadVariableOp%conv_net/conv2d/Conv2D/ReadVariableOp2T
(conv_net/conv2d_1/BiasAdd/ReadVariableOp(conv_net/conv2d_1/BiasAdd/ReadVariableOp2R
'conv_net/conv2d_1/Conv2D/ReadVariableOp'conv_net/conv2d_1/Conv2D/ReadVariableOp2T
(conv_net/conv2d_2/BiasAdd/ReadVariableOp(conv_net/conv2d_2/BiasAdd/ReadVariableOp2R
'conv_net/conv2d_2/Conv2D/ReadVariableOp'conv_net/conv2d_2/Conv2D/ReadVariableOp2T
(conv_net/conv2d_3/BiasAdd/ReadVariableOp(conv_net/conv2d_3/BiasAdd/ReadVariableOp2R
'conv_net/conv2d_3/Conv2D/ReadVariableOp'conv_net/conv2d_3/Conv2D/ReadVariableOp2T
(conv_net/conv2d_4/BiasAdd/ReadVariableOp(conv_net/conv2d_4/BiasAdd/ReadVariableOp2R
'conv_net/conv2d_4/Conv2D/ReadVariableOp'conv_net/conv2d_4/Conv2D/ReadVariableOp2T
(conv_net/conv2d_5/BiasAdd/ReadVariableOp(conv_net/conv2d_5/BiasAdd/ReadVariableOp2R
'conv_net/conv2d_5/Conv2D/ReadVariableOp'conv_net/conv2d_5/Conv2D/ReadVariableOp2N
%conv_net/dense/BiasAdd/ReadVariableOp%conv_net/dense/BiasAdd/ReadVariableOp2L
$conv_net/dense/MatMul/ReadVariableOp$conv_net/dense/MatMul/ReadVariableOp2R
'conv_net/dense_1/BiasAdd/ReadVariableOp'conv_net/dense_1/BiasAdd/ReadVariableOp2P
&conv_net/dense_1/MatMul/ReadVariableOp&conv_net/dense_1/MatMul/ReadVariableOp:' #
!
_user_specified_name	input_1
╩	
┌
A__inference_dense_layer_call_and_return_conditional_losses_573760

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АHА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         АH::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
зу
ь
"__inference__traced_restore_574545
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rate-
)assignvariableop_5_conv_net_conv2d_kernel+
'assignvariableop_6_conv_net_conv2d_bias/
+assignvariableop_7_conv_net_conv2d_1_kernel-
)assignvariableop_8_conv_net_conv2d_1_bias/
+assignvariableop_9_conv_net_conv2d_2_kernel.
*assignvariableop_10_conv_net_conv2d_2_bias0
,assignvariableop_11_conv_net_conv2d_3_kernel.
*assignvariableop_12_conv_net_conv2d_3_bias0
,assignvariableop_13_conv_net_conv2d_4_kernel.
*assignvariableop_14_conv_net_conv2d_4_bias0
,assignvariableop_15_conv_net_conv2d_5_kernel.
*assignvariableop_16_conv_net_conv2d_5_bias-
)assignvariableop_17_conv_net_dense_kernel+
'assignvariableop_18_conv_net_dense_bias/
+assignvariableop_19_conv_net_dense_1_kernel-
)assignvariableop_20_conv_net_dense_1_bias
assignvariableop_21_total
assignvariableop_22_count5
1assignvariableop_23_adam_conv_net_conv2d_kernel_m3
/assignvariableop_24_adam_conv_net_conv2d_bias_m7
3assignvariableop_25_adam_conv_net_conv2d_1_kernel_m5
1assignvariableop_26_adam_conv_net_conv2d_1_bias_m7
3assignvariableop_27_adam_conv_net_conv2d_2_kernel_m5
1assignvariableop_28_adam_conv_net_conv2d_2_bias_m7
3assignvariableop_29_adam_conv_net_conv2d_3_kernel_m5
1assignvariableop_30_adam_conv_net_conv2d_3_bias_m7
3assignvariableop_31_adam_conv_net_conv2d_4_kernel_m5
1assignvariableop_32_adam_conv_net_conv2d_4_bias_m7
3assignvariableop_33_adam_conv_net_conv2d_5_kernel_m5
1assignvariableop_34_adam_conv_net_conv2d_5_bias_m4
0assignvariableop_35_adam_conv_net_dense_kernel_m2
.assignvariableop_36_adam_conv_net_dense_bias_m6
2assignvariableop_37_adam_conv_net_dense_1_kernel_m4
0assignvariableop_38_adam_conv_net_dense_1_bias_m5
1assignvariableop_39_adam_conv_net_conv2d_kernel_v3
/assignvariableop_40_adam_conv_net_conv2d_bias_v7
3assignvariableop_41_adam_conv_net_conv2d_1_kernel_v5
1assignvariableop_42_adam_conv_net_conv2d_1_bias_v7
3assignvariableop_43_adam_conv_net_conv2d_2_kernel_v5
1assignvariableop_44_adam_conv_net_conv2d_2_bias_v7
3assignvariableop_45_adam_conv_net_conv2d_3_kernel_v5
1assignvariableop_46_adam_conv_net_conv2d_3_bias_v7
3assignvariableop_47_adam_conv_net_conv2d_4_kernel_v5
1assignvariableop_48_adam_conv_net_conv2d_4_bias_v7
3assignvariableop_49_adam_conv_net_conv2d_5_kernel_v5
1assignvariableop_50_adam_conv_net_conv2d_5_bias_v4
0assignvariableop_51_adam_conv_net_dense_kernel_v2
.assignvariableop_52_adam_conv_net_dense_bias_v6
2assignvariableop_53_adam_conv_net_dense_1_kernel_v4
0assignvariableop_54_adam_conv_net_dense_1_bias_v
identity_56ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_53вAssignVariableOp_54вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9в	RestoreV2вRestoreV2_1╠
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*╪
value╬B╦7B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names¤
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*Б
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices┴
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Є
_output_shapes▀
▄:::::::::::::::::::::::::::::::::::::::::::::::::::::::*E
dtypes;
927	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0	*
_output_shapes
:2

IdentityК
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Ф
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2Ф
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3У
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4Ы
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5Я
AssignVariableOp_5AssignVariableOp)assignvariableop_5_conv_net_conv2d_kernelIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6Э
AssignVariableOp_6AssignVariableOp'assignvariableop_6_conv_net_conv2d_biasIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7б
AssignVariableOp_7AssignVariableOp+assignvariableop_7_conv_net_conv2d_1_kernelIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8Я
AssignVariableOp_8AssignVariableOp)assignvariableop_8_conv_net_conv2d_1_biasIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9б
AssignVariableOp_9AssignVariableOp+assignvariableop_9_conv_net_conv2d_2_kernelIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10г
AssignVariableOp_10AssignVariableOp*assignvariableop_10_conv_net_conv2d_2_biasIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11е
AssignVariableOp_11AssignVariableOp,assignvariableop_11_conv_net_conv2d_3_kernelIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12г
AssignVariableOp_12AssignVariableOp*assignvariableop_12_conv_net_conv2d_3_biasIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13е
AssignVariableOp_13AssignVariableOp,assignvariableop_13_conv_net_conv2d_4_kernelIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14г
AssignVariableOp_14AssignVariableOp*assignvariableop_14_conv_net_conv2d_4_biasIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15е
AssignVariableOp_15AssignVariableOp,assignvariableop_15_conv_net_conv2d_5_kernelIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16г
AssignVariableOp_16AssignVariableOp*assignvariableop_16_conv_net_conv2d_5_biasIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17в
AssignVariableOp_17AssignVariableOp)assignvariableop_17_conv_net_dense_kernelIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18а
AssignVariableOp_18AssignVariableOp'assignvariableop_18_conv_net_dense_biasIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19д
AssignVariableOp_19AssignVariableOp+assignvariableop_19_conv_net_dense_1_kernelIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20в
AssignVariableOp_20AssignVariableOp)assignvariableop_20_conv_net_dense_1_biasIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21Т
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22Т
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23к
AssignVariableOp_23AssignVariableOp1assignvariableop_23_adam_conv_net_conv2d_kernel_mIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24и
AssignVariableOp_24AssignVariableOp/assignvariableop_24_adam_conv_net_conv2d_bias_mIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25м
AssignVariableOp_25AssignVariableOp3assignvariableop_25_adam_conv_net_conv2d_1_kernel_mIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26к
AssignVariableOp_26AssignVariableOp1assignvariableop_26_adam_conv_net_conv2d_1_bias_mIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27м
AssignVariableOp_27AssignVariableOp3assignvariableop_27_adam_conv_net_conv2d_2_kernel_mIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28к
AssignVariableOp_28AssignVariableOp1assignvariableop_28_adam_conv_net_conv2d_2_bias_mIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29м
AssignVariableOp_29AssignVariableOp3assignvariableop_29_adam_conv_net_conv2d_3_kernel_mIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30к
AssignVariableOp_30AssignVariableOp1assignvariableop_30_adam_conv_net_conv2d_3_bias_mIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31м
AssignVariableOp_31AssignVariableOp3assignvariableop_31_adam_conv_net_conv2d_4_kernel_mIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32к
AssignVariableOp_32AssignVariableOp1assignvariableop_32_adam_conv_net_conv2d_4_bias_mIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33м
AssignVariableOp_33AssignVariableOp3assignvariableop_33_adam_conv_net_conv2d_5_kernel_mIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34к
AssignVariableOp_34AssignVariableOp1assignvariableop_34_adam_conv_net_conv2d_5_bias_mIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35й
AssignVariableOp_35AssignVariableOp0assignvariableop_35_adam_conv_net_dense_kernel_mIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36з
AssignVariableOp_36AssignVariableOp.assignvariableop_36_adam_conv_net_dense_bias_mIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37л
AssignVariableOp_37AssignVariableOp2assignvariableop_37_adam_conv_net_dense_1_kernel_mIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38й
AssignVariableOp_38AssignVariableOp0assignvariableop_38_adam_conv_net_dense_1_bias_mIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39к
AssignVariableOp_39AssignVariableOp1assignvariableop_39_adam_conv_net_conv2d_kernel_vIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40и
AssignVariableOp_40AssignVariableOp/assignvariableop_40_adam_conv_net_conv2d_bias_vIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41м
AssignVariableOp_41AssignVariableOp3assignvariableop_41_adam_conv_net_conv2d_1_kernel_vIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42к
AssignVariableOp_42AssignVariableOp1assignvariableop_42_adam_conv_net_conv2d_1_bias_vIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43м
AssignVariableOp_43AssignVariableOp3assignvariableop_43_adam_conv_net_conv2d_2_kernel_vIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44к
AssignVariableOp_44AssignVariableOp1assignvariableop_44_adam_conv_net_conv2d_2_bias_vIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45м
AssignVariableOp_45AssignVariableOp3assignvariableop_45_adam_conv_net_conv2d_3_kernel_vIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46к
AssignVariableOp_46AssignVariableOp1assignvariableop_46_adam_conv_net_conv2d_3_bias_vIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47м
AssignVariableOp_47AssignVariableOp3assignvariableop_47_adam_conv_net_conv2d_4_kernel_vIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48к
AssignVariableOp_48AssignVariableOp1assignvariableop_48_adam_conv_net_conv2d_4_bias_vIdentity_48:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:2
Identity_49м
AssignVariableOp_49AssignVariableOp3assignvariableop_49_adam_conv_net_conv2d_5_kernel_vIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50к
AssignVariableOp_50AssignVariableOp1assignvariableop_50_adam_conv_net_conv2d_5_bias_vIdentity_50:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_50_
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:2
Identity_51й
AssignVariableOp_51AssignVariableOp0assignvariableop_51_adam_conv_net_dense_kernel_vIdentity_51:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_51_
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:2
Identity_52з
AssignVariableOp_52AssignVariableOp.assignvariableop_52_adam_conv_net_dense_bias_vIdentity_52:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_52_
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:2
Identity_53л
AssignVariableOp_53AssignVariableOp2assignvariableop_53_adam_conv_net_dense_1_kernel_vIdentity_53:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_53_
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:2
Identity_54й
AssignVariableOp_54AssignVariableOp0assignvariableop_54_adam_conv_net_dense_1_bias_vIdentity_54:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_54и
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesФ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices─
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpШ

Identity_55Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_55е

Identity_56IdentityIdentity_55:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_56"#
identity_56Identity_56:output:0*є
_input_shapesс
▐: :::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
К
_
C__inference_flatten_layer_call_and_return_conditional_losses_573741

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"     $  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         АH2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         АH2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:& "
 
_user_specified_nameinputs
ш
▌
D__inference_conv2d_3_layer_call_and_return_conditional_losses_573649

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp╡
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
Еi
ъ
__inference__traced_save_574368
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop5
1savev2_conv_net_conv2d_kernel_read_readvariableop3
/savev2_conv_net_conv2d_bias_read_readvariableop7
3savev2_conv_net_conv2d_1_kernel_read_readvariableop5
1savev2_conv_net_conv2d_1_bias_read_readvariableop7
3savev2_conv_net_conv2d_2_kernel_read_readvariableop5
1savev2_conv_net_conv2d_2_bias_read_readvariableop7
3savev2_conv_net_conv2d_3_kernel_read_readvariableop5
1savev2_conv_net_conv2d_3_bias_read_readvariableop7
3savev2_conv_net_conv2d_4_kernel_read_readvariableop5
1savev2_conv_net_conv2d_4_bias_read_readvariableop7
3savev2_conv_net_conv2d_5_kernel_read_readvariableop5
1savev2_conv_net_conv2d_5_bias_read_readvariableop4
0savev2_conv_net_dense_kernel_read_readvariableop2
.savev2_conv_net_dense_bias_read_readvariableop6
2savev2_conv_net_dense_1_kernel_read_readvariableop4
0savev2_conv_net_dense_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_adam_conv_net_conv2d_kernel_m_read_readvariableop:
6savev2_adam_conv_net_conv2d_bias_m_read_readvariableop>
:savev2_adam_conv_net_conv2d_1_kernel_m_read_readvariableop<
8savev2_adam_conv_net_conv2d_1_bias_m_read_readvariableop>
:savev2_adam_conv_net_conv2d_2_kernel_m_read_readvariableop<
8savev2_adam_conv_net_conv2d_2_bias_m_read_readvariableop>
:savev2_adam_conv_net_conv2d_3_kernel_m_read_readvariableop<
8savev2_adam_conv_net_conv2d_3_bias_m_read_readvariableop>
:savev2_adam_conv_net_conv2d_4_kernel_m_read_readvariableop<
8savev2_adam_conv_net_conv2d_4_bias_m_read_readvariableop>
:savev2_adam_conv_net_conv2d_5_kernel_m_read_readvariableop<
8savev2_adam_conv_net_conv2d_5_bias_m_read_readvariableop;
7savev2_adam_conv_net_dense_kernel_m_read_readvariableop9
5savev2_adam_conv_net_dense_bias_m_read_readvariableop=
9savev2_adam_conv_net_dense_1_kernel_m_read_readvariableop;
7savev2_adam_conv_net_dense_1_bias_m_read_readvariableop<
8savev2_adam_conv_net_conv2d_kernel_v_read_readvariableop:
6savev2_adam_conv_net_conv2d_bias_v_read_readvariableop>
:savev2_adam_conv_net_conv2d_1_kernel_v_read_readvariableop<
8savev2_adam_conv_net_conv2d_1_bias_v_read_readvariableop>
:savev2_adam_conv_net_conv2d_2_kernel_v_read_readvariableop<
8savev2_adam_conv_net_conv2d_2_bias_v_read_readvariableop>
:savev2_adam_conv_net_conv2d_3_kernel_v_read_readvariableop<
8savev2_adam_conv_net_conv2d_3_bias_v_read_readvariableop>
:savev2_adam_conv_net_conv2d_4_kernel_v_read_readvariableop<
8savev2_adam_conv_net_conv2d_4_bias_v_read_readvariableop>
:savev2_adam_conv_net_conv2d_5_kernel_v_read_readvariableop<
8savev2_adam_conv_net_conv2d_5_bias_v_read_readvariableop;
7savev2_adam_conv_net_dense_kernel_v_read_readvariableop9
5savev2_adam_conv_net_dense_bias_v_read_readvariableop=
9savev2_adam_conv_net_dense_1_kernel_v_read_readvariableop;
7savev2_adam_conv_net_dense_1_bias_v_read_readvariableop
savev2_1_const

identity_1ИвMergeV2CheckpointsвSaveV2вSaveV2_1е
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_97e14ce8e2c34cd0b36c304b8cce032b/part2
StringJoin/inputs_1Б

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename╞
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*╪
value╬B╦7B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesў
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*Б
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesю
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop1savev2_conv_net_conv2d_kernel_read_readvariableop/savev2_conv_net_conv2d_bias_read_readvariableop3savev2_conv_net_conv2d_1_kernel_read_readvariableop1savev2_conv_net_conv2d_1_bias_read_readvariableop3savev2_conv_net_conv2d_2_kernel_read_readvariableop1savev2_conv_net_conv2d_2_bias_read_readvariableop3savev2_conv_net_conv2d_3_kernel_read_readvariableop1savev2_conv_net_conv2d_3_bias_read_readvariableop3savev2_conv_net_conv2d_4_kernel_read_readvariableop1savev2_conv_net_conv2d_4_bias_read_readvariableop3savev2_conv_net_conv2d_5_kernel_read_readvariableop1savev2_conv_net_conv2d_5_bias_read_readvariableop0savev2_conv_net_dense_kernel_read_readvariableop.savev2_conv_net_dense_bias_read_readvariableop2savev2_conv_net_dense_1_kernel_read_readvariableop0savev2_conv_net_dense_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_adam_conv_net_conv2d_kernel_m_read_readvariableop6savev2_adam_conv_net_conv2d_bias_m_read_readvariableop:savev2_adam_conv_net_conv2d_1_kernel_m_read_readvariableop8savev2_adam_conv_net_conv2d_1_bias_m_read_readvariableop:savev2_adam_conv_net_conv2d_2_kernel_m_read_readvariableop8savev2_adam_conv_net_conv2d_2_bias_m_read_readvariableop:savev2_adam_conv_net_conv2d_3_kernel_m_read_readvariableop8savev2_adam_conv_net_conv2d_3_bias_m_read_readvariableop:savev2_adam_conv_net_conv2d_4_kernel_m_read_readvariableop8savev2_adam_conv_net_conv2d_4_bias_m_read_readvariableop:savev2_adam_conv_net_conv2d_5_kernel_m_read_readvariableop8savev2_adam_conv_net_conv2d_5_bias_m_read_readvariableop7savev2_adam_conv_net_dense_kernel_m_read_readvariableop5savev2_adam_conv_net_dense_bias_m_read_readvariableop9savev2_adam_conv_net_dense_1_kernel_m_read_readvariableop7savev2_adam_conv_net_dense_1_bias_m_read_readvariableop8savev2_adam_conv_net_conv2d_kernel_v_read_readvariableop6savev2_adam_conv_net_conv2d_bias_v_read_readvariableop:savev2_adam_conv_net_conv2d_1_kernel_v_read_readvariableop8savev2_adam_conv_net_conv2d_1_bias_v_read_readvariableop:savev2_adam_conv_net_conv2d_2_kernel_v_read_readvariableop8savev2_adam_conv_net_conv2d_2_bias_v_read_readvariableop:savev2_adam_conv_net_conv2d_3_kernel_v_read_readvariableop8savev2_adam_conv_net_conv2d_3_bias_v_read_readvariableop:savev2_adam_conv_net_conv2d_4_kernel_v_read_readvariableop8savev2_adam_conv_net_conv2d_4_bias_v_read_readvariableop:savev2_adam_conv_net_conv2d_5_kernel_v_read_readvariableop8savev2_adam_conv_net_conv2d_5_bias_v_read_readvariableop7savev2_adam_conv_net_dense_kernel_v_read_readvariableop5savev2_adam_conv_net_dense_bias_v_read_readvariableop9savev2_adam_conv_net_dense_1_kernel_v_read_readvariableop7savev2_adam_conv_net_dense_1_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *E
dtypes;
927	2
SaveV2Г
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardм
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1в
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesО
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices╧
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1у
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesм
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityБ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*├
_input_shapes▒
о: : : : : : ::::: : :  : : @:@:@@:@:
АHА:А:	А:: : ::::: : :  : : @:@:@@:@:
АHА:А:	А:::::: : :  : : @:@:@@:@:
АHА:А:	А:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
╦
L
0__inference_max_pooling2d_1_layer_call_fn_573669

inputs
identity╓
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4                                    **
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_5736632
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
ч6
║
D__inference_conv_net_layer_call_and_return_conditional_losses_573827
input_1)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2+
'conv2d_5_statefulpartitionedcall_args_1+
'conv2d_5_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallв conv2d_3/StatefulPartitionedCallв conv2d_4/StatefulPartitionedCallв conv2d_5/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallй
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         22**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_5735742 
conv2d/StatefulPartitionedCall╙
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         22**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_5735952"
 conv2d_1/StatefulPartitionedCall°
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_5736092
max_pooling2d/PartitionedCall╥
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:          **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_5736282"
 conv2d_2/StatefulPartitionedCall╒
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:          **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_5736492"
 conv2d_3/StatefulPartitionedCall■
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:          **
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_5736632!
max_pooling2d_1/PartitionedCall╘
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_5736822"
 conv2d_4/StatefulPartitionedCall╒
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0'conv2d_5_statefulpartitionedcall_args_1'conv2d_5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_5737032"
 conv2d_5/StatefulPartitionedCall▀
flatten/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         АH**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_5737412
flatten/PartitionedCall╢
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_5737602
dense/StatefulPartitionedCall┼
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_5737832!
dense_1/StatefulPartitionedCallО
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:         22::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
╬	
▄
C__inference_dense_1_layer_call_and_return_conditional_losses_573783

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
╟
J
.__inference_max_pooling2d_layer_call_fn_573615

inputs
identity╘
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4                                    **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_5736092
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
ў
д
)__inference_conv_net_layer_call_fn_574111
x"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16
identityИвStatefulPartitionedCall╓
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv_net_layer_call_and_return_conditional_losses_5738612
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:         22::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
ш
▌
D__inference_conv2d_2_layer_call_and_return_conditional_losses_573628

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp╡
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
╒6
┤
D__inference_conv_net_layer_call_and_return_conditional_losses_573913
x)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2+
'conv2d_5_statefulpartitionedcall_args_1+
'conv2d_5_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallв conv2d_3/StatefulPartitionedCallв conv2d_4/StatefulPartitionedCallв conv2d_5/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallг
conv2d/StatefulPartitionedCallStatefulPartitionedCallx%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         22**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_5735742 
conv2d/StatefulPartitionedCall╙
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         22**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_5735952"
 conv2d_1/StatefulPartitionedCall°
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_5736092
max_pooling2d/PartitionedCall╥
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:          **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_5736282"
 conv2d_2/StatefulPartitionedCall╒
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:          **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_5736492"
 conv2d_3/StatefulPartitionedCall■
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:          **
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_5736632!
max_pooling2d_1/PartitionedCall╘
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_5736822"
 conv2d_4/StatefulPartitionedCall╒
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0'conv2d_5_statefulpartitionedcall_args_1'conv2d_5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_5737032"
 conv2d_5/StatefulPartitionedCall▀
flatten/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         АH**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_5737412
flatten/PartitionedCall╢
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_5737602
dense/StatefulPartitionedCall┼
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_5737832!
dense_1/StatefulPartitionedCallО
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:         22::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:! 

_user_specified_namex"пL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*│
serving_defaultЯ
C
input_18
serving_default_input_1:0         22<
output_10
StatefulPartitionedCall:0         tensorflow/serving/predict:·П
В
sequence
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
+░&call_and_return_all_conditional_losses
▒_default_save_signature
▓__call__"ж
_tf_keras_modelМ{"class_name": "ConvNet", "name": "conv_net", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "is_graph_network": false, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "ConvNet"}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
n
0
	1

2
3
4
5
6
7
8
9
10"
trackable_list_wrapper
У
iter

beta_1

beta_2
	decay
learning_ratemРmСmТmУmФmХmЦmЧ mШ!mЩ"mЪ#mЫ$mЬ%mЭ&mЮ'mЯvаvбvвvгvдvеvжvз vи!vй"vк#vл$vм%vн&vо'vп"
	optimizer
Ц
0
1
2
3
4
5
6
7
 8
!9
"10
#11
$12
%13
&14
'15"
trackable_list_wrapper
Ц
0
1
2
3
4
5
6
7
 8
!9
"10
#11
$12
%13
&14
'15"
trackable_list_wrapper
 "
trackable_list_wrapper
╗
(metrics
trainable_variables
)layer_regularization_losses
	variables

*layers
+non_trainable_variables
regularization_losses
▓__call__
▒_default_save_signature
+░&call_and_return_all_conditional_losses
'░"call_and_return_conditional_losses"
_generic_user_object
-
│serving_default"
signature_map
щ

kernel
bias
,trainable_variables
-	variables
.regularization_losses
/	keras_api
+┤&call_and_return_all_conditional_losses
╡__call__"┬
_tf_keras_layerи{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
ю

kernel
bias
0trainable_variables
1	variables
2regularization_losses
3	keras_api
+╢&call_and_return_all_conditional_losses
╖__call__"╟
_tf_keras_layerн{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}}
√
4trainable_variables
5	variables
6regularization_losses
7	keras_api
+╕&call_and_return_all_conditional_losses
╣__call__"ъ
_tf_keras_layer╨{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ю

kernel
bias
8trainable_variables
9	variables
:regularization_losses
;	keras_api
+║&call_and_return_all_conditional_losses
╗__call__"╟
_tf_keras_layerн{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}}
ю

kernel
bias
<trainable_variables
=	variables
>regularization_losses
?	keras_api
+╝&call_and_return_all_conditional_losses
╜__call__"╟
_tf_keras_layerн{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
 
@trainable_variables
A	variables
Bregularization_losses
C	keras_api
+╛&call_and_return_all_conditional_losses
┐__call__"ю
_tf_keras_layer╘{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ю

 kernel
!bias
Dtrainable_variables
E	variables
Fregularization_losses
G	keras_api
+└&call_and_return_all_conditional_losses
┴__call__"╟
_tf_keras_layerн{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
ю

"kernel
#bias
Htrainable_variables
I	variables
Jregularization_losses
K	keras_api
+┬&call_and_return_all_conditional_losses
├__call__"╟
_tf_keras_layerн{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
о
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
+─&call_and_return_all_conditional_losses
┼__call__"Э
_tf_keras_layerГ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
Є

$kernel
%bias
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api
+╞&call_and_return_all_conditional_losses
╟__call__"╦
_tf_keras_layer▒{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 9216}}}}
Ў

&kernel
'bias
Ttrainable_variables
U	variables
Vregularization_losses
W	keras_api
+╚&call_and_return_all_conditional_losses
╔__call__"╧
_tf_keras_layer╡{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
0:.2conv_net/conv2d/kernel
": 2conv_net/conv2d/bias
2:02conv_net/conv2d_1/kernel
$:"2conv_net/conv2d_1/bias
2:0 2conv_net/conv2d_2/kernel
$:" 2conv_net/conv2d_2/bias
2:0  2conv_net/conv2d_3/kernel
$:" 2conv_net/conv2d_3/bias
2:0 @2conv_net/conv2d_4/kernel
$:"@2conv_net/conv2d_4/bias
2:0@@2conv_net/conv2d_5/kernel
$:"@2conv_net/conv2d_5/bias
):'
АHА2conv_net/dense/kernel
": А2conv_net/dense/bias
*:(	А2conv_net/dense_1/kernel
#:!2conv_net/dense_1/bias
'
X0"
trackable_list_wrapper
 "
trackable_list_wrapper
n
0
	1

2
3
4
5
6
7
8
9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Э
Ymetrics
,trainable_variables
Zlayer_regularization_losses
-	variables

[layers
\non_trainable_variables
.regularization_losses
╡__call__
+┤&call_and_return_all_conditional_losses
'┤"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Э
]metrics
0trainable_variables
^layer_regularization_losses
1	variables

_layers
`non_trainable_variables
2regularization_losses
╖__call__
+╢&call_and_return_all_conditional_losses
'╢"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Э
ametrics
4trainable_variables
blayer_regularization_losses
5	variables

clayers
dnon_trainable_variables
6regularization_losses
╣__call__
+╕&call_and_return_all_conditional_losses
'╕"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Э
emetrics
8trainable_variables
flayer_regularization_losses
9	variables

glayers
hnon_trainable_variables
:regularization_losses
╗__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Э
imetrics
<trainable_variables
jlayer_regularization_losses
=	variables

klayers
lnon_trainable_variables
>regularization_losses
╜__call__
+╝&call_and_return_all_conditional_losses
'╝"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Э
mmetrics
@trainable_variables
nlayer_regularization_losses
A	variables

olayers
pnon_trainable_variables
Bregularization_losses
┐__call__
+╛&call_and_return_all_conditional_losses
'╛"call_and_return_conditional_losses"
_generic_user_object
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
Э
qmetrics
Dtrainable_variables
rlayer_regularization_losses
E	variables

slayers
tnon_trainable_variables
Fregularization_losses
┴__call__
+└&call_and_return_all_conditional_losses
'└"call_and_return_conditional_losses"
_generic_user_object
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
Э
umetrics
Htrainable_variables
vlayer_regularization_losses
I	variables

wlayers
xnon_trainable_variables
Jregularization_losses
├__call__
+┬&call_and_return_all_conditional_losses
'┬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Э
ymetrics
Ltrainable_variables
zlayer_regularization_losses
M	variables

{layers
|non_trainable_variables
Nregularization_losses
┼__call__
+─&call_and_return_all_conditional_losses
'─"call_and_return_conditional_losses"
_generic_user_object
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ю
}metrics
Ptrainable_variables
~layer_regularization_losses
Q	variables

layers
Аnon_trainable_variables
Rregularization_losses
╟__call__
+╞&call_and_return_all_conditional_losses
'╞"call_and_return_conditional_losses"
_generic_user_object
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
б
Бmetrics
Ttrainable_variables
 Вlayer_regularization_losses
U	variables
Гlayers
Дnon_trainable_variables
Vregularization_losses
╔__call__
+╚&call_and_return_all_conditional_losses
'╚"call_and_return_conditional_losses"
_generic_user_object
г

Еtotal

Жcount
З
_fn_kwargs
Иtrainable_variables
Й	variables
Кregularization_losses
Л	keras_api
+╩&call_and_return_all_conditional_losses
╦__call__"х
_tf_keras_layer╦{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
Е0
Ж1"
trackable_list_wrapper
 "
trackable_list_wrapper
д
Мmetrics
Иtrainable_variables
 Нlayer_regularization_losses
Й	variables
Оlayers
Пnon_trainable_variables
Кregularization_losses
╦__call__
+╩&call_and_return_all_conditional_losses
'╩"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Е0
Ж1"
trackable_list_wrapper
5:32Adam/conv_net/conv2d/kernel/m
':%2Adam/conv_net/conv2d/bias/m
7:52Adam/conv_net/conv2d_1/kernel/m
):'2Adam/conv_net/conv2d_1/bias/m
7:5 2Adam/conv_net/conv2d_2/kernel/m
):' 2Adam/conv_net/conv2d_2/bias/m
7:5  2Adam/conv_net/conv2d_3/kernel/m
):' 2Adam/conv_net/conv2d_3/bias/m
7:5 @2Adam/conv_net/conv2d_4/kernel/m
):'@2Adam/conv_net/conv2d_4/bias/m
7:5@@2Adam/conv_net/conv2d_5/kernel/m
):'@2Adam/conv_net/conv2d_5/bias/m
.:,
АHА2Adam/conv_net/dense/kernel/m
':%А2Adam/conv_net/dense/bias/m
/:-	А2Adam/conv_net/dense_1/kernel/m
(:&2Adam/conv_net/dense_1/bias/m
5:32Adam/conv_net/conv2d/kernel/v
':%2Adam/conv_net/conv2d/bias/v
7:52Adam/conv_net/conv2d_1/kernel/v
):'2Adam/conv_net/conv2d_1/bias/v
7:5 2Adam/conv_net/conv2d_2/kernel/v
):' 2Adam/conv_net/conv2d_2/bias/v
7:5  2Adam/conv_net/conv2d_3/kernel/v
):' 2Adam/conv_net/conv2d_3/bias/v
7:5 @2Adam/conv_net/conv2d_4/kernel/v
):'@2Adam/conv_net/conv2d_4/bias/v
7:5@@2Adam/conv_net/conv2d_5/kernel/v
):'@2Adam/conv_net/conv2d_5/bias/v
.:,
АHА2Adam/conv_net/dense/kernel/v
':%А2Adam/conv_net/dense/bias/v
/:-	А2Adam/conv_net/dense_1/kernel/v
(:&2Adam/conv_net/dense_1/bias/v
┘2╓
D__inference_conv_net_layer_call_and_return_conditional_losses_574026
D__inference_conv_net_layer_call_and_return_conditional_losses_574090
D__inference_conv_net_layer_call_and_return_conditional_losses_573827
D__inference_conv_net_layer_call_and_return_conditional_losses_573796╗
▓▓о
FullArgSpec,
args$Ъ!
jself
jx

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ч2ф
!__inference__wrapped_model_573561╛
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *.в+
)К&
input_1         22
э2ъ
)__inference_conv_net_layer_call_fn_573932
)__inference_conv_net_layer_call_fn_574111
)__inference_conv_net_layer_call_fn_573880
)__inference_conv_net_layer_call_fn_574132╗
▓▓о
FullArgSpec,
args$Ъ!
jself
jx

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
3B1
$__inference_signature_wrapper_573962input_1
б2Ю
B__inference_conv2d_layer_call_and_return_conditional_losses_573574╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
Ж2Г
'__inference_conv2d_layer_call_fn_573582╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
г2а
D__inference_conv2d_1_layer_call_and_return_conditional_losses_573595╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
И2Е
)__inference_conv2d_1_layer_call_fn_573603╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
▒2о
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_573609р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ц2У
.__inference_max_pooling2d_layer_call_fn_573615р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
г2а
D__inference_conv2d_2_layer_call_and_return_conditional_losses_573628╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
И2Е
)__inference_conv2d_2_layer_call_fn_573636╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
г2а
D__inference_conv2d_3_layer_call_and_return_conditional_losses_573649╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
И2Е
)__inference_conv2d_3_layer_call_fn_573657╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
│2░
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_573663р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ш2Х
0__inference_max_pooling2d_1_layer_call_fn_573669р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
г2а
D__inference_conv2d_4_layer_call_and_return_conditional_losses_573682╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
И2Е
)__inference_conv2d_4_layer_call_fn_573690╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
г2а
D__inference_conv2d_5_layer_call_and_return_conditional_losses_573703╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           @
И2Е
)__inference_conv2d_5_layer_call_fn_573711╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           @
э2ъ
C__inference_flatten_layer_call_and_return_conditional_losses_574138в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_flatten_layer_call_fn_574143в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_dense_layer_call_and_return_conditional_losses_574154в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╨2═
&__inference_dense_layer_call_fn_574161в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_dense_1_layer_call_and_return_conditional_losses_574172в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense_1_layer_call_fn_574179в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 з
!__inference__wrapped_model_573561Б !"#$%&'8в5
.в+
)К&
input_1         22
к "3к0
.
output_1"К
output_1         ┘
D__inference_conv2d_1_layer_call_and_return_conditional_losses_573595РIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ ▒
)__inference_conv2d_1_layer_call_fn_573603ГIвF
?в<
:К7
inputs+                           
к "2К/+                           ┘
D__inference_conv2d_2_layer_call_and_return_conditional_losses_573628РIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                            
Ъ ▒
)__inference_conv2d_2_layer_call_fn_573636ГIвF
?в<
:К7
inputs+                           
к "2К/+                            ┘
D__inference_conv2d_3_layer_call_and_return_conditional_losses_573649РIвF
?в<
:К7
inputs+                            
к "?в<
5К2
0+                            
Ъ ▒
)__inference_conv2d_3_layer_call_fn_573657ГIвF
?в<
:К7
inputs+                            
к "2К/+                            ┘
D__inference_conv2d_4_layer_call_and_return_conditional_losses_573682Р !IвF
?в<
:К7
inputs+                            
к "?в<
5К2
0+                           @
Ъ ▒
)__inference_conv2d_4_layer_call_fn_573690Г !IвF
?в<
:К7
inputs+                            
к "2К/+                           @┘
D__inference_conv2d_5_layer_call_and_return_conditional_losses_573703Р"#IвF
?в<
:К7
inputs+                           @
к "?в<
5К2
0+                           @
Ъ ▒
)__inference_conv2d_5_layer_call_fn_573711Г"#IвF
?в<
:К7
inputs+                           @
к "2К/+                           @╫
B__inference_conv2d_layer_call_and_return_conditional_losses_573574РIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ п
'__inference_conv2d_layer_call_fn_573582ГIвF
?в<
:К7
inputs+                           
к "2К/+                           ├
D__inference_conv_net_layer_call_and_return_conditional_losses_573796{ !"#$%&'@в=
6в3
)К&
input_1         22
p

 
к "%в"
К
0         
Ъ ├
D__inference_conv_net_layer_call_and_return_conditional_losses_573827{ !"#$%&'@в=
6в3
)К&
input_1         22
p 

 
к "%в"
К
0         
Ъ ╜
D__inference_conv_net_layer_call_and_return_conditional_losses_574026u !"#$%&':в7
0в-
#К 
x         22
p

 
к "%в"
К
0         
Ъ ╜
D__inference_conv_net_layer_call_and_return_conditional_losses_574090u !"#$%&':в7
0в-
#К 
x         22
p 

 
к "%в"
К
0         
Ъ Ы
)__inference_conv_net_layer_call_fn_573880n !"#$%&'@в=
6в3
)К&
input_1         22
p

 
к "К         Ы
)__inference_conv_net_layer_call_fn_573932n !"#$%&'@в=
6в3
)К&
input_1         22
p 

 
к "К         Х
)__inference_conv_net_layer_call_fn_574111h !"#$%&':в7
0в-
#К 
x         22
p

 
к "К         Х
)__inference_conv_net_layer_call_fn_574132h !"#$%&':в7
0в-
#К 
x         22
p 

 
к "К         д
C__inference_dense_1_layer_call_and_return_conditional_losses_574172]&'0в-
&в#
!К
inputs         А
к "%в"
К
0         
Ъ |
(__inference_dense_1_layer_call_fn_574179P&'0в-
&в#
!К
inputs         А
к "К         г
A__inference_dense_layer_call_and_return_conditional_losses_574154^$%0в-
&в#
!К
inputs         АH
к "&в#
К
0         А
Ъ {
&__inference_dense_layer_call_fn_574161Q$%0в-
&в#
!К
inputs         АH
к "К         Аи
C__inference_flatten_layer_call_and_return_conditional_losses_574138a7в4
-в*
(К%
inputs         @
к "&в#
К
0         АH
Ъ А
(__inference_flatten_layer_call_fn_574143T7в4
-в*
(К%
inputs         @
к "К         АHю
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_573663ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╞
0__inference_max_pooling2d_1_layer_call_fn_573669СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ь
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_573609ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ─
.__inference_max_pooling2d_layer_call_fn_573615СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ╡
$__inference_signature_wrapper_573962М !"#$%&'Cв@
в 
9к6
4
input_1)К&
input_1         22"3к0
.
output_1"К
output_1         