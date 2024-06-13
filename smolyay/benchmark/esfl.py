import numpy

from .benchmark import BenchmarkFunction


class esfl(BenchmarkFunction):
    @property
    def domain(self):
        return [[-9.5195173415, 9.43243439265], [-9.5099228378, 9.44106944598]]

    @property
    def global_minimum(self):
        return 191.223912413
    
    @property
    def global_minimum_location(self):
        return [0.4804826585, 0.4900771622]

    def _function(self, x):
        y = (
            0.357488738
            * numpy.sqrt(
                numpy.square((-0.171747132) + x[..., 0]) + numpy.square((-0.570083725) + x[..., 1])
            )
            + 0.022353275
            * numpy.sqrt(
                numpy.square((-0.843266708) + x[..., 0]) + numpy.square((-0.11103765) + x[..., 1])
            )
            + 0.979447225
            * numpy.sqrt(
                numpy.square((-0.550375356) + x[..., 0]) + numpy.square((-0.987006267) + x[..., 1])
            )
            + 0.765630995
            * numpy.sqrt(
                numpy.square((-0.301137904) + x[..., 0]) + numpy.square((-0.519399582) + x[..., 1])
            )
            + 0.023184715
            * numpy.sqrt(
                numpy.square((-0.292212117) + x[..., 0]) + numpy.square((-0.706486263) + x[..., 1])
            )
            + 0.844444149
            * numpy.sqrt(
                numpy.square((-0.224052867) + x[..., 0]) + numpy.square((-0.842017851) + x[..., 1])
            )
            + 0.215317421
            * numpy.sqrt(
                numpy.square((-0.349830504) + x[..., 0]) + numpy.square((-0.657379219) + x[..., 1])
            )
            + 0.16940845
            * numpy.sqrt(
                numpy.square((-0.856270347) + x[..., 0]) + numpy.square((-0.116512821) + x[..., 1])
            )
            + 0.306907123
            * numpy.sqrt(
                numpy.square((-0.067113723) + x[..., 0]) + numpy.square((-0.274247976) + x[..., 1])
            )
            + 0.087539484
            * numpy.sqrt(
                numpy.square((-0.500210669) + x[..., 0]) + numpy.square((-0.083160206) + x[..., 1])
            )
            + 0.381054289
            * numpy.sqrt(
                numpy.square((-0.998117627) + x[..., 0]) + numpy.square((-0.505309967) + x[..., 1])
            )
            + 0.213595635
            * numpy.sqrt(
                numpy.square((-0.578733378) + x[..., 0]) + numpy.square((-0.232684875) + x[..., 1])
            )
        )
        y1 = (
            0.898815006
            * numpy.sqrt(
                numpy.square((-0.991133039) + x[..., 0]) + numpy.square((-0.798570132) + x[..., 1])
            )
            + 0.288096197
            * numpy.sqrt(
                numpy.square((-0.762250467) + x[..., 0]) + numpy.square((-0.988052693) + x[..., 1])
            )
            + 0.294916264
            * numpy.sqrt(
                numpy.square((-0.130692483) + x[..., 0]) + numpy.square((-0.325052691) + x[..., 1])
            )
            + 0.796987249
            * numpy.sqrt(
                numpy.square((-0.639718759) + x[..., 0]) + numpy.square((-0.855372665) + x[..., 1])
            )
            + 0.151058462
            * numpy.sqrt(
                numpy.square((-0.159517864) + x[..., 0]) + numpy.square((-0.789978159) + x[..., 1])
            )
            + 0.180359679
            * numpy.sqrt(
                numpy.square((-0.250080533) + x[..., 0]) + numpy.square((-0.321093678) + x[..., 1])
            )
            + 0.358008111
            * numpy.sqrt(
                numpy.square((-0.668928609) + x[..., 0]) + numpy.square((-0.212870341) + x[..., 1])
            )
            + 0.292473004
            * numpy.sqrt(
                numpy.square((-0.435356381) + x[..., 0]) + numpy.square((-0.520887305) + x[..., 1])
            )
            + 0.345182256
            * numpy.sqrt(
                numpy.square((-0.359700266) + x[..., 0]) + numpy.square((-0.340500054) + x[..., 1])
            )
            + 0.795001604
            * numpy.sqrt(
                numpy.square((-0.351441368) + x[..., 0]) + numpy.square((-0.256409548) + x[..., 1])
            )
            + 0.01232667
            * numpy.sqrt(
                numpy.square((-0.13149159) + x[..., 0]) + numpy.square((-0.993065527) + x[..., 1])
            )
            + 0.339135032
            * numpy.sqrt(
                numpy.square((-0.150101788) + x[..., 0]) + numpy.square((-0.247789113) + x[..., 1])
            )
        )
        y2 = (
            0.814449473
            * numpy.sqrt(
                numpy.square((-0.58911365) + x[..., 0]) + numpy.square((-0.121480362) + x[..., 1])
            )
            + 0.846893965
            * numpy.sqrt(
                numpy.square((-0.830892812) + x[..., 0]) + numpy.square((-0.943705036) + x[..., 1])
            )
            + 0.247020813
            * numpy.sqrt(
                numpy.square((-0.230815738) + x[..., 0]) + numpy.square((-0.25319562) + x[..., 1])
            )
            + 0.279295622
            * numpy.sqrt(
                numpy.square((-0.66573446) + x[..., 0]) + numpy.square((-0.363348714) + x[..., 1])
            )
            + 0.368279721
            * numpy.sqrt(
                numpy.square((-0.775857606) + x[..., 0]) + numpy.square((-0.394098723) + x[..., 1])
            )
            + 0.646594313
            * numpy.sqrt(
                numpy.square((-0.303658477) + x[..., 0]) + numpy.square((-0.836343458) + x[..., 1])
            )
            + 0.583919624
            * numpy.sqrt(
                numpy.square((-0.110492291) + x[..., 0]) + numpy.square((-0.443998928) + x[..., 1])
            )
            + 0.316718743
            * numpy.sqrt(
                numpy.square((-0.502384866) + x[..., 0]) + numpy.square((-0.926295884) + x[..., 1])
            )
            + 0.240831638
            * numpy.sqrt(
                numpy.square((-0.160172762) + x[..., 0]) + numpy.square((-0.479206379) + x[..., 1])
            )
            + 0.281779784
            * numpy.sqrt(
                numpy.square((-0.872462311) + x[..., 0]) + numpy.square((-0.903877983) + x[..., 1])
            )
            + 0.930871765
            * numpy.sqrt(
                numpy.square((-0.265114545) + x[..., 0]) + numpy.square((-0.509528966) + x[..., 1])
            )
            + 0.264077204
            * numpy.sqrt(
                numpy.square((-0.285814322) + x[..., 0]) + numpy.square((-0.914176571) + x[..., 1])
            )
        )
        y3 = (
            0.587996037
            * numpy.sqrt(
                numpy.square((-0.593955922) + x[..., 0]) + numpy.square((-0.192409508) + x[..., 1])
            )
            + 0.476536165
            * numpy.sqrt(
                numpy.square((-0.722719071) + x[..., 0]) + numpy.square((-0.144140655) + x[..., 1])
            )
            + 0.424643255
            * numpy.sqrt(
                numpy.square((-0.628248677) + x[..., 0]) + numpy.square((-0.760373813) + x[..., 1])
            )
            + 0.4863258
            * numpy.sqrt(
                numpy.square((-0.463797865) + x[..., 0]) + numpy.square((-0.330575992) + x[..., 1])
            )
            + 0.645501823
            * numpy.sqrt(
                numpy.square((-0.413306994) + x[..., 0]) + numpy.square((-0.000185665) + x[..., 1])
            )
            + 0.500974464
            * numpy.sqrt(
                numpy.square((-0.117695357) + x[..., 0]) + numpy.square((-0.54342633) + x[..., 1])
            )
            + 0.018404303
            * numpy.sqrt(
                numpy.square((-0.314212267) + x[..., 0]) + numpy.square((-0.535061193) + x[..., 1])
            )
            + 0.545260953
            * numpy.sqrt(
                numpy.square((-0.046551514) + x[..., 0]) + numpy.square((-0.868599462) + x[..., 1])
            )
            + 0.490911826
            * numpy.sqrt(
                numpy.square((-0.338550272) + x[..., 0]) + numpy.square((-0.45453827) + x[..., 1])
            )
            + 0.675539251
            * numpy.sqrt(
                numpy.square((-0.182099593) + x[..., 0]) + numpy.square((-0.358328183) + x[..., 1])
            )
            + 0.118965818
            * numpy.sqrt(
                numpy.square((-0.645727127) + x[..., 0]) + numpy.square((-0.1138901) + x[..., 1])
            )
            + 0.369012282
            * numpy.sqrt(
                numpy.square((-0.560745547) + x[..., 0]) + numpy.square((-0.669544787) + x[..., 1])
            )
        )
        y4 = (
            0.798181177
            * numpy.sqrt(
                numpy.square((-0.76996172) + x[..., 0]) + numpy.square((-0.938007998) + x[..., 1])
            )
            + 0.23807268
            * numpy.sqrt(
                numpy.square((-0.297805864) + x[..., 0]) + numpy.square((-0.015072031) + x[..., 1])
            )
            + 0.729639041
            * numpy.sqrt(
                numpy.square((-0.661106261) + x[..., 0]) + numpy.square((-0.341922751) + x[..., 1])
            )
            + 0.656389222
            * numpy.sqrt(
                numpy.square((-0.755821674) + x[..., 0]) + numpy.square((-0.865366275) + x[..., 1])
            )
            + 0.619633636
            * numpy.sqrt(
                numpy.square((-0.627447499) + x[..., 0]) + numpy.square((-0.460852668) + x[..., 1])
            )
            + 0.193344047
            * numpy.sqrt(
                numpy.square((-0.283864198) + x[..., 0]) + numpy.square((-0.421057594) + x[..., 1])
            )
            + 0.64839479
            * numpy.sqrt(
                numpy.square((-0.086424624) + x[..., 0]) + numpy.square((-0.725831669) + x[..., 1])
            )
            + 0.804792138
            * numpy.sqrt(
                numpy.square((-0.102514669) + x[..., 0]) + numpy.square((-0.669588676) + x[..., 1])
            )
            + 0.537523786
            * numpy.sqrt(
                numpy.square((-0.641251151) + x[..., 0]) + numpy.square((-0.092898668) + x[..., 1])
            )
            + 0.960780581
            * numpy.sqrt(
                numpy.square((-0.545309498) + x[..., 0]) + numpy.square((-0.526591373) + x[..., 1])
            )
            + 0.715000616
            * numpy.sqrt(
                numpy.square((-0.031524852) + x[..., 0]) + numpy.square((-0.89543162) + x[..., 1])
            )
            + 0.733906419
            * numpy.sqrt(
                numpy.square((-0.792360642) + x[..., 0]) + numpy.square((-0.990129874) + x[..., 1])
            )
            + 0.062322046
            * numpy.sqrt(
                numpy.square((-0.072766998) + x[..., 0]) + numpy.square((-0.506862765) + x[..., 1])
            )
            + 0.529565623
            * numpy.sqrt(
                numpy.square((-0.175661049) + x[..., 0]) + numpy.square((-0.644503586) + x[..., 1])
            )
            + 0.210178445
            * numpy.sqrt(
                numpy.square((-0.525632613) + x[..., 0]) + numpy.square((-0.760300662) + x[..., 1])
            )
            + 0.08842876
            * numpy.sqrt(
                numpy.square((-0.750207669) + x[..., 0]) + numpy.square((-0.906079247) + x[..., 1])
            )
            + 0.785206925
            * numpy.sqrt(
                numpy.square((-0.178123714) + x[..., 0]) + numpy.square((-0.16628849) + x[..., 1])
            )
            + 0.215813519
            * numpy.sqrt(
                numpy.square((-0.034140986) + x[..., 0]) + numpy.square((-0.515180583) + x[..., 1])
            )
            + 0.215612012
            * numpy.sqrt(
                numpy.square((-0.585131173) + x[..., 0]) + numpy.square((-0.024994567) + x[..., 1])
            )
            + 0.248496923
            * numpy.sqrt(
                numpy.square((-0.621229984) + x[..., 0]) + numpy.square((-0.448178475) + x[..., 1])
            )
            + 0.281994409
            * numpy.sqrt(
                numpy.square((-0.3893619) + x[..., 0]) + numpy.square((-0.501291257) + x[..., 1])
            )
            + 0.660094564
            * numpy.sqrt(
                numpy.square((-0.358714153) + x[..., 0]) + numpy.square((-0.681191699) + x[..., 1])
            )
            + 0.222232689
            * numpy.sqrt(
                numpy.square((-0.243034617) + x[..., 0]) + numpy.square((-0.799044649) + x[..., 1])
            )
            + 0.720349714
            * numpy.sqrt(
                numpy.square((-0.246421539) + x[..., 0]) + numpy.square((-0.8729527) + x[..., 1])
            )
            + 0.463571696
            * numpy.sqrt(
                numpy.square((-0.130502803) + x[..., 0]) + numpy.square((-0.282977315) + x[..., 1])
            )
            + 0.123991796
            * numpy.sqrt(
                numpy.square((-0.93344972) + x[..., 0]) + numpy.square((-0.910494023) + x[..., 1])
            )
            + 0.498715128
            * numpy.sqrt(
                numpy.square((-0.379937906) + x[..., 0]) + numpy.square((-0.450857975) + x[..., 1])
            )
            + 0.572276021
            * numpy.sqrt(
                numpy.square((-0.783400461) + x[..., 0]) + numpy.square((-0.874014477) + x[..., 1])
            )
            + 0.459367789
            * numpy.sqrt(
                numpy.square((-0.300034258) + x[..., 0]) + numpy.square((-0.216889666) + x[..., 1])
            )
            + 0.671128163
            * numpy.sqrt(
                numpy.square((-0.125483222) + x[..., 0]) + numpy.square((-0.080130778) + x[..., 1])
            )
            + 0.249773036
            * numpy.sqrt(
                numpy.square((-0.748874105) + x[..., 0]) + numpy.square((-0.705291626) + x[..., 1])
            )
            + 0.5526966
            * numpy.sqrt(
                numpy.square((-0.069232463) + x[..., 0]) + numpy.square((-0.900495049) + x[..., 1])
            )
            + 0.484829489
            * numpy.sqrt(
                numpy.square((-0.202015557) + x[..., 0]) + numpy.square((-0.018138982) + x[..., 1])
            )
            + 0.018666644
            * numpy.sqrt(
                numpy.square((-0.005065858) + x[..., 0]) + numpy.square((-0.460414894) + x[..., 1])
            )
            + 0.050630379
            * numpy.sqrt(
                numpy.square((-0.269613052) + x[..., 0]) + numpy.square((-0.623967962) + x[..., 1])
            )
            + 0.289278296
            * numpy.sqrt(
                numpy.square((-0.499851475) + x[..., 0]) + numpy.square((-0.716356389) + x[..., 1])
            )
            + 0.782122103
            * numpy.sqrt(
                numpy.square((-0.151285869) + x[..., 0]) + numpy.square((-0.335155086) + x[..., 1])
            )
            + 0.685751798
            * numpy.sqrt(
                numpy.square((-0.174169455) + x[..., 0]) + numpy.square((-0.012875633) + x[..., 1])
            )
            + 0.959230005
            * numpy.sqrt(
                numpy.square((-0.330637734) + x[..., 0]) + numpy.square((-0.356212159) + x[..., 1])
            )
            + 0.218478363
            * numpy.sqrt(
                numpy.square((-0.316906054) + x[..., 0]) + numpy.square((-0.368168729) + x[..., 1])
            )
            + 0.302332559
            * numpy.sqrt(
                numpy.square((-0.322086955) + x[..., 0]) + numpy.square((-0.916871716) + x[..., 1])
            )
            + 0.16524077
            * numpy.sqrt(
                numpy.square((-0.963976641) + x[..., 0]) + numpy.square((-0.990129384) + x[..., 1])
            )
            + 0.199815445
            * numpy.sqrt(
                numpy.square((-0.993602205) + x[..., 0]) + numpy.square((-0.207690308) + x[..., 1])
            )
            + 0.114452047
            * numpy.sqrt(
                numpy.square((-0.369903055) + x[..., 0]) + numpy.square((-0.350391657) + x[..., 1])
            )
            + 0.98432755
            * numpy.sqrt(
                numpy.square((-0.372888567) + x[..., 0]) + numpy.square((-0.486761436) + x[..., 1])
            )
            + 0.368902677
            * numpy.sqrt(
                numpy.square((-0.77197833) + x[..., 0]) + numpy.square((-0.643860992) + x[..., 1])
            )
            + 0.078851617
            * numpy.sqrt(
                numpy.square((-0.396684142) + x[..., 0]) + numpy.square((-0.056328016) + x[..., 1])
            )
            + 0.201846876
            * numpy.sqrt(
                numpy.square((-0.913096325) + x[..., 0]) + numpy.square((-0.917025459) + x[..., 1])
            )
            + 0.735401099
            * numpy.sqrt(
                numpy.square((-0.11957773) + x[..., 0]) + numpy.square((-0.038116363) + x[..., 1])
            )
            + 0.066869495
            * numpy.sqrt(
                numpy.square((-0.735478889) + x[..., 0]) + numpy.square((-0.302376318) + x[..., 1])
            )
            + 0.884760067
            * numpy.sqrt(
                numpy.square((-0.055418475) + x[..., 0]) + numpy.square((-0.07002933) + x[..., 1])
            )
            + 0.685453582
            * numpy.sqrt(
                numpy.square((-0.576299805) + x[..., 0]) + numpy.square((-0.466485577) + x[..., 1])
            )
            + 0.440087942
            * numpy.sqrt(
                numpy.square((-0.05140711) + x[..., 0]) + numpy.square((-0.560696814) + x[..., 1])
            )
            + 0.383058929
            * numpy.sqrt(
                numpy.square((-0.006008368) + x[..., 0]) + numpy.square((-0.46509422) + x[..., 1])
            )
            + 0.294164391
            * numpy.sqrt(
                numpy.square((-0.401227683) + x[..., 0]) + numpy.square((-0.547329056) + x[..., 1])
            )
            + 0.835583945
            * numpy.sqrt(
                numpy.square((-0.519881187) + x[..., 0]) + numpy.square((-0.177966463) + x[..., 1])
            )
            + 0.132485187
            * numpy.sqrt(
                numpy.square((-0.628877255) + x[..., 0]) + numpy.square((-0.381959759) + x[..., 1])
            )
            + 0.776184866
            * numpy.sqrt(
                numpy.square((-0.22574988) + x[..., 0]) + numpy.square((-0.512915631) + x[..., 1])
            )
            + 0.575025027
            * numpy.sqrt(
                numpy.square((-0.396121408) + x[..., 0]) + numpy.square((-0.956546118) + x[..., 1])
            )
            + 0.835805098
            * numpy.sqrt(
                numpy.square((-0.276006131) + x[..., 0]) + numpy.square((-0.602291185) + x[..., 1])
            )
            + 0.641181702
            * numpy.sqrt(
                numpy.square((-0.152372608) + x[..., 0]) + numpy.square((-0.454816916) + x[..., 1])
            )
            + 0.767634036
            * numpy.sqrt(
                numpy.square((-0.936322836) + x[..., 0]) + numpy.square((-0.69008483) + x[..., 1])
            )
            + 0.050276715
            * numpy.sqrt(
                numpy.square((-0.42266059) + x[..., 0]) + numpy.square((-0.707101625) + x[..., 1])
            )
            + 0.290766977
            * numpy.sqrt(
                numpy.square((-0.134663129) + x[..., 0]) + numpy.square((-0.089831717) + x[..., 1])
            )
            + 0.671752978
            * numpy.sqrt(
                numpy.square((-0.386055614) + x[..., 0]) + numpy.square((-0.40809923) + x[..., 1])
            )
            + 0.320759459
            * numpy.sqrt(
                numpy.square((-0.374632747) + x[..., 0]) + numpy.square((-0.136829933) + x[..., 1])
            )
            + 0.064261759
            * numpy.sqrt(
                numpy.square((-0.26848104) + x[..., 0]) + numpy.square((-0.734626153) + x[..., 1])
            )
            + 0.47354399
            * numpy.sqrt(
                numpy.square((-0.948370515) + x[..., 0]) + numpy.square((-0.592765273) + x[..., 1])
            )
            + 0.492208615
            * numpy.sqrt(
                numpy.square((-0.188940325) + x[..., 0]) + numpy.square((-0.357459622) + x[..., 1])
            )
            + 0.055740578
            * numpy.sqrt(
                numpy.square((-0.297509548) + x[..., 0]) + numpy.square((-0.073677179) + x[..., 1])
            )
            + 0.407474183
            * numpy.sqrt(
                numpy.square((-0.074552766) + x[..., 0]) + numpy.square((-0.630809186) + x[..., 1])
            )
            + 0.443654947
            * numpy.sqrt(
                numpy.square((-0.401346257) + x[..., 0]) + numpy.square((-0.776008147) + x[..., 1])
            )
            + 0.765573365
            * numpy.sqrt(
                numpy.square((-0.101689197) + x[..., 0]) + numpy.square((-0.420482819) + x[..., 1])
            )
            + 0.951534808
            * numpy.sqrt(
                numpy.square((-0.38388961) + x[..., 0]) + numpy.square((-0.807434133) + x[..., 1])
            )
            + 0.660500886
            * numpy.sqrt(
                numpy.square((-0.324093927) + x[..., 0]) + numpy.square((-0.604649583) + x[..., 1])
            )
            + 0.805458244
            * numpy.sqrt(
                numpy.square((-0.192134382) + x[..., 0]) + numpy.square((-0.659033407) + x[..., 1])
            )
            + 0.235451309
            * numpy.sqrt(
                numpy.square((-0.112368436) + x[..., 0]) + numpy.square((-0.797631369) + x[..., 1])
            )
            + 0.735906889
            * numpy.sqrt(
                numpy.square((-0.596558144) + x[..., 0]) + numpy.square((-0.64879588) + x[..., 1])
            )
            + 0.074847891
            * numpy.sqrt(
                numpy.square((-0.511448928) + x[..., 0]) + numpy.square((-0.122215731) + x[..., 1])
            )
            + 0.962597232
            * numpy.sqrt(
                numpy.square((-0.045066059) + x[..., 0]) + numpy.square((-0.760041535) + x[..., 1])
            )
            + 0.105587493
            * numpy.sqrt(
                numpy.square((-0.783102004) + x[..., 0]) + numpy.square((-0.633112272) + x[..., 1])
            )
            + 0.953454175
            * numpy.sqrt(
                numpy.square((-0.945749415) + x[..., 0]) + numpy.square((-0.82900712) + x[..., 1])
            )
            + 0.081364195
            * numpy.sqrt(
                numpy.square((-0.596462556) + x[..., 0]) + numpy.square((-0.908820157) + x[..., 1])
            )
            + 0.783828538
            * numpy.sqrt(
                numpy.square((-0.607341301) + x[..., 0]) + numpy.square((-0.858561483) + x[..., 1])
            )
            + 0.552162345
            * numpy.sqrt(
                numpy.square((-0.362509471) + x[..., 0]) + numpy.square((-0.966240678) + x[..., 1])
            )
            + 0.880760754
            * numpy.sqrt(
                numpy.square((-0.594067961) + x[..., 0]) + numpy.square((-0.035746839) + x[..., 1])
            )
            + 0.754515423
            * numpy.sqrt(
                numpy.square((-0.679854079) + x[..., 0]) + numpy.square((-0.962487051) + x[..., 1])
            )
            + 0.246756809
            * numpy.sqrt(
                numpy.square((-0.506588022) + x[..., 0]) + numpy.square((-0.003066951) + x[..., 1])
            )
            + 0.289027603
            * numpy.sqrt(
                numpy.square((-0.159253884) + x[..., 0]) + numpy.square((-0.118492143) + x[..., 1])
            )
            + 0.394241157
            * numpy.sqrt(
                numpy.square((-0.656892105) + x[..., 0]) + numpy.square((-0.758601687) + x[..., 1])
            )
            + 0.66964466
            * numpy.sqrt(
                numpy.square((-0.523879602) + x[..., 0]) + numpy.square((-0.255503721) + x[..., 1])
            )
            + 0.588778056
            * numpy.sqrt(
                numpy.square((-0.124396483) + x[..., 0]) + numpy.square((-0.914097492) + x[..., 1])
            )
            + 0.037357008
            * numpy.sqrt(
                numpy.square((-0.986720724) + x[..., 0]) + numpy.square((-0.287043964) + x[..., 1])
            )
            + 0.154437867
            * numpy.sqrt(
                numpy.square((-0.228123065) + x[..., 0]) + numpy.square((-0.686623483) + x[..., 1])
            )
            + 0.680954577
            * numpy.sqrt(
                numpy.square((-0.675654903) + x[..., 0]) + numpy.square((-0.275270061) + x[..., 1])
            )
            + 0.341551978
            * numpy.sqrt(
                numpy.square((-0.776777457) + x[..., 0]) + numpy.square((-0.390280343) + x[..., 1])
            )
            + 0.450240154
            * numpy.sqrt(
                numpy.square((-0.932451789) + x[..., 0]) + numpy.square((-0.094697764) + x[..., 1])
            )
            + 0.969393248
            * numpy.sqrt(
                numpy.square((-0.201241563) + x[..., 0]) + numpy.square((-0.217560434) + x[..., 1])
            )
            + 0.588504411
            * numpy.sqrt(
                numpy.square((-0.297136057) + x[..., 0]) + numpy.square((-0.843528892) + x[..., 1])
            )
            + 0.761841907
            * numpy.sqrt(
                numpy.square((-0.197227518) + x[..., 0]) + numpy.square((-0.84225785) + x[..., 1])
            )
            + 0.417323251
            * numpy.sqrt(
                numpy.square((-0.246345717) + x[..., 0]) + numpy.square((-0.88356033) + x[..., 1])
            )
            + 0.061565034
            * numpy.sqrt(
                numpy.square((-0.646476473) + x[..., 0]) + numpy.square((-0.150248769) + x[..., 1])
            )
            + 0.427990929
            * numpy.sqrt(
                numpy.square((-0.734972611) + x[..., 0]) + numpy.square((-0.750736969) + x[..., 1])
            )
            + 0.099308429
            * numpy.sqrt(
                numpy.square((-0.085436744) + x[..., 0]) + numpy.square((-0.52293578) + x[..., 1])
            )
            + 0.315150132
            * numpy.sqrt(
                numpy.square((-0.150347716) + x[..., 0]) + numpy.square((-0.277381751) + x[..., 1])
            )
            + 0.338711228
            * numpy.sqrt(
                numpy.square((-0.434188491) + x[..., 0]) + numpy.square((-0.621850855) + x[..., 1])
            )
            + 0.864322888
            * numpy.sqrt(
                numpy.square((-0.186937905) + x[..., 0]) + numpy.square((-0.96519432) + x[..., 1])
            )
            + 0.57353931
            * numpy.sqrt(
                numpy.square((-0.692692957) + x[..., 0]) + numpy.square((-0.572948596) + x[..., 1])
            )
            + 0.726003657
            * numpy.sqrt(
                numpy.square((-0.762973751) + x[..., 0]) + numpy.square((-0.335737923) + x[..., 1])
            )
            + 0.421163376
            * numpy.sqrt(
                numpy.square((-0.154806144) + x[..., 0]) + numpy.square((-0.399842147) + x[..., 1])
            )
            + 0.608387812
            * numpy.sqrt(
                numpy.square((-0.389378384) + x[..., 0]) + numpy.square((-0.520377889) + x[..., 1])
            )
            + 0.093767189
            * numpy.sqrt(
                numpy.square((-0.695427535) + x[..., 0]) + numpy.square((-0.218795105) + x[..., 1])
            )
        )
        y5 = (
            0.707490758
            * numpy.sqrt(
                numpy.square((-0.845811974) + x[..., 0]) + numpy.square((-0.701109783) + x[..., 1])
            )
            + 0.179114414
            * numpy.sqrt(
                numpy.square((-0.612720947) + x[..., 0]) + numpy.square((-0.353886874) + x[..., 1])
            )
            + 0.492064502
            * numpy.sqrt(
                numpy.square((-0.975971873) + x[..., 0]) + numpy.square((-0.304426516) + x[..., 1])
            )
            + 0.472078286
            * numpy.sqrt(
                numpy.square((-0.026889386) + x[..., 0]) + numpy.square((-0.953670458) + x[..., 1])
            )
            + 0.181037673
            * numpy.sqrt(
                numpy.square((-0.187448731) + x[..., 0]) + numpy.square((-0.201667047) + x[..., 1])
            )
            + 0.131345237
            * numpy.sqrt(
                numpy.square((-0.087118836) + x[..., 0]) + numpy.square((-0.922581037) + x[..., 1])
            )
            + 0.176495355
            * numpy.sqrt(
                numpy.square((-0.540400638) + x[..., 0]) + numpy.square((-0.468724525) + x[..., 1])
            )
            + 0.158910241
            * numpy.sqrt(
                numpy.square((-0.126864289) + x[..., 0]) + numpy.square((-0.802522718) + x[..., 1])
            )
            + 0.501543329
            * numpy.sqrt(
                numpy.square((-0.733999033) + x[..., 0]) + numpy.square((-0.334774364) + x[..., 1])
            )
            + 0.230422197
            * numpy.sqrt(
                numpy.square((-0.11323201) + x[..., 0]) + numpy.square((-0.081413617) + x[..., 1])
            )
            + 0.525947636
            * numpy.sqrt(
                numpy.square((-0.488353947) + x[..., 0]) + numpy.square((-0.133484697) + x[..., 1])
            )
            + 0.222485643
            * numpy.sqrt(
                numpy.square((-0.795600371) + x[..., 0]) + numpy.square((-0.760500488) + x[..., 1])
            )
            + 0.951579426
            * numpy.sqrt(
                numpy.square((-0.492047073) + x[..., 0]) + numpy.square((-0.094155321) + x[..., 1])
            )
            + 0.017286583
            * numpy.sqrt(
                numpy.square((-0.533560992) + x[..., 0]) + numpy.square((-0.445370842) + x[..., 1])
            )
            + 0.773855947
            * numpy.sqrt(
                numpy.square((-0.010624414) + x[..., 0]) + numpy.square((-0.793827034) + x[..., 1])
            )
            + 0.63609267
            * numpy.sqrt(
                numpy.square((-0.543870155) + x[..., 0]) + numpy.square((-0.462614988) + x[..., 1])
            )
            + 0.751719363
            * numpy.sqrt(
                numpy.square((-0.451129087) + x[..., 0]) + numpy.square((-0.324499852) + x[..., 1])
            )
            + 0.546048839
            * numpy.sqrt(
                numpy.square((-0.975328385) + x[..., 0]) + numpy.square((-0.353885964) + x[..., 1])
            )
            + 0.821130694
            * numpy.sqrt(
                numpy.square((-0.183847189) + x[..., 0]) + numpy.square((-0.174235179) + x[..., 1])
            )
            + 0.866548632
            * numpy.sqrt(
                numpy.square((-0.163532267) + x[..., 0]) + numpy.square((-0.987748218) + x[..., 1])
            )
            + 0.657055272
            * numpy.sqrt(
                numpy.square((-0.024634437) + x[..., 0]) + numpy.square((-0.129356297) + x[..., 1])
            )
            + 0.873001015
            * numpy.sqrt(
                numpy.square((-0.177822574) + x[..., 0]) + numpy.square((-0.147486909) + x[..., 1])
            )
            + 0.227196133
            * numpy.sqrt(
                numpy.square((-0.061318512) + x[..., 0]) + numpy.square((-0.657588574) + x[..., 1])
            )
            + 0.038717998
            * numpy.sqrt(
                numpy.square((-0.016643898) + x[..., 0]) + numpy.square((-0.163164654) + x[..., 1])
            )
            + 0.068694108
            * numpy.sqrt(
                numpy.square((-0.835654851) + x[..., 0]) + numpy.square((-0.811805848) + x[..., 1])
            )
            + 0.395910613
            * numpy.sqrt(
                numpy.square((-0.601659033) + x[..., 0]) + numpy.square((-0.602659244) + x[..., 1])
            )
            + 0.343740596
            * numpy.sqrt(
                numpy.square((-0.02701678) + x[..., 0]) + numpy.square((-0.297864669) + x[..., 1])
            )
            + 0.169103416
            * numpy.sqrt(
                numpy.square((-0.196093864) + x[..., 0]) + numpy.square((-0.501146458) + x[..., 1])
            )
            + 0.586596363
            * numpy.sqrt(
                numpy.square((-0.950710745) + x[..., 0]) + numpy.square((-0.488417783) + x[..., 1])
            )
            + 0.918931478
            * numpy.sqrt(
                numpy.square((-0.335541754) + x[..., 0]) + numpy.square((-0.784520588) + x[..., 1])
            )
            + 0.114271622
            * numpy.sqrt(
                numpy.square((-0.594262491) + x[..., 0]) + numpy.square((-0.621107192) + x[..., 1])
            )
            + 0.512842737
            * numpy.sqrt(
                numpy.square((-0.259191325) + x[..., 0]) + numpy.square((-0.605576512) + x[..., 1])
            )
            + 0.81921613
            * numpy.sqrt(
                numpy.square((-0.640633714) + x[..., 0]) + numpy.square((-0.802055569) + x[..., 1])
            )
            + 0.833839562
            * numpy.sqrt(
                numpy.square((-0.15524903) + x[..., 0]) + numpy.square((-0.653212536) + x[..., 1])
            )
            + 0.947596354
            * numpy.sqrt(
                numpy.square((-0.460016568) + x[..., 0]) + numpy.square((-0.459280576) + x[..., 1])
            )
            + 0.269722972
            * numpy.sqrt(
                numpy.square((-0.393339954) + x[..., 0]) + numpy.square((-0.596959134) + x[..., 1])
            )
            + 0.104161152
            * numpy.sqrt(
                numpy.square((-0.805462475) + x[..., 0]) + numpy.square((-0.979521858) + x[..., 1])
            )
            + 0.3901876
            * numpy.sqrt(
                numpy.square((-0.540991774) + x[..., 0]) + numpy.square((-0.185438344) + x[..., 1])
            )
            + 0.669799
            * numpy.sqrt(
                numpy.square((-0.390721843) + x[..., 0]) + numpy.square((-0.982181113) + x[..., 1])
            )
            + 0.129480695
            * numpy.sqrt(
                numpy.square((-0.557819042) + x[..., 0]) + numpy.square((-0.957867831) + x[..., 1])
            )
            + 0.080746111
            * numpy.sqrt(
                numpy.square((-0.932760523) + x[..., 0]) + numpy.square((-0.482934281) + x[..., 1])
            )
            + 0.538074843
            * numpy.sqrt(
                numpy.square((-0.348765542) + x[..., 0]) + numpy.square((-0.250763619) + x[..., 1])
            )
            + 0.628041563
            * numpy.sqrt(
                numpy.square((-0.008287193) + x[..., 0]) + numpy.square((-0.484798228) + x[..., 1])
            )
            + 0.43855974
            * numpy.sqrt(
                numpy.square((-0.948836169) + x[..., 0]) + numpy.square((-0.809883151) + x[..., 1])
            )
            + 0.257970696
            * numpy.sqrt(
                numpy.square((-0.571923707) + x[..., 0]) + numpy.square((-0.519439583) + x[..., 1])
            )
            + 0.165023741
            * numpy.sqrt(
                numpy.square((-0.333626354) + x[..., 0]) + numpy.square((-0.974728822) + x[..., 1])
            )
            + 0.091153125
            * numpy.sqrt(
                numpy.square((-0.983747547) + x[..., 0]) + numpy.square((-0.102452961) + x[..., 1])
            )
            + 0.532824017
            * numpy.sqrt(
                numpy.square((-0.766458106) + x[..., 0]) + numpy.square((-0.469947618) + x[..., 1])
            )
            + 0.3811668
            * numpy.sqrt(
                numpy.square((-0.110094564) + x[..., 0]) + numpy.square((-0.017201272) + x[..., 1])
            )
            + 0.805034101
            * numpy.sqrt(
                numpy.square((-0.994803523) + x[..., 0]) + numpy.square((-0.306160913) + x[..., 1])
            )
            + 0.124313929
            * numpy.sqrt(
                numpy.square((-0.580324521) + x[..., 0]) + numpy.square((-0.300928208) + x[..., 1])
            )
            + 0.130513419
            * numpy.sqrt(
                numpy.square((-0.166415607) + x[..., 0]) + numpy.square((-0.280450367) + x[..., 1])
            )
            + 0.97615329
            * numpy.sqrt(
                numpy.square((-0.643357216) + x[..., 0]) + numpy.square((-0.433257026) + x[..., 1])
            )
            + 0.480500108
            * numpy.sqrt(
                numpy.square((-0.344312325) + x[..., 0]) + numpy.square((-0.502627288) + x[..., 1])
            )
            + 0.370199323
            * numpy.sqrt(
                numpy.square((-0.912325531) + x[..., 0]) + numpy.square((-0.201078217) + x[..., 1])
            )
            + 0.977507878
            * numpy.sqrt(
                numpy.square((-0.900062559) + x[..., 0]) + numpy.square((-0.552619243) + x[..., 1])
            )
            + 0.647526689
            * numpy.sqrt(
                numpy.square((-0.016258391) + x[..., 0]) + numpy.square((-0.883153364) + x[..., 1])
            )
            + 0.721863216
            * numpy.sqrt(
                numpy.square((-0.368630572) + x[..., 0]) + numpy.square((-0.579510443) + x[..., 1])
            )
            + 0.028978527
            * numpy.sqrt(
                numpy.square((-0.664379915) + x[..., 0]) + numpy.square((-0.788261923) + x[..., 1])
            )
            + 0.015604686
            * numpy.sqrt(
                numpy.square((-0.59338077) + x[..., 0]) + numpy.square((-0.113091411) + x[..., 1])
            )
            + 0.820801571
            * numpy.sqrt(
                numpy.square((-0.034570895) + x[..., 0]) + numpy.square((-0.658438338) + x[..., 1])
            )
            + 0.155990922
            * numpy.sqrt(
                numpy.square((-0.84181978) + x[..., 0]) + numpy.square((-0.066412438) + x[..., 1])
            )
            + 0.353558262
            * numpy.sqrt(
                numpy.square((-0.932080672) + x[..., 0]) + numpy.square((-0.898055989) + x[..., 1])
            )
            + 0.861441613
            * numpy.sqrt(
                numpy.square((-0.507964677) + x[..., 0]) + numpy.square((-0.886605669) + x[..., 1])
            )
            + 0.888647137
            * numpy.sqrt(
                numpy.square((-0.299596772) + x[..., 0]) + numpy.square((-0.732627597) + x[..., 1])
            )
            + 0.050429568
            * numpy.sqrt(
                numpy.square((-0.49662343) + x[..., 0]) + numpy.square((-0.627874579) + x[..., 1])
            )
            + 0.133800344
            * numpy.sqrt(
                numpy.square((-0.044930442) + x[..., 0]) + numpy.square((-0.365208401) + x[..., 1])
            )
            + 0.978763511
            * numpy.sqrt(
                numpy.square((-0.773703434) + x[..., 0]) + numpy.square((-0.0372145) + x[..., 1])
            )
            + 0.803594338
            * numpy.sqrt(
                numpy.square((-0.532969856) + x[..., 0]) + numpy.square((-0.728657023) + x[..., 1])
            )
            + 0.262967188
            * numpy.sqrt(
                numpy.square((-0.74676686) + x[..., 0]) + numpy.square((-0.614168639) + x[..., 1])
            )
            + 0.008155304
            * numpy.sqrt(
                numpy.square((-0.720050146) + x[..., 0]) + numpy.square((-0.839521521) + x[..., 1])
            )
            + 0.782326115
            * numpy.sqrt(
                numpy.square((-0.631600574) + x[..., 0]) + numpy.square((-0.922979188) + x[..., 1])
            )
            + 0.957544428
            * numpy.sqrt(
                numpy.square((-0.11491679) + x[..., 0]) + numpy.square((-0.236607042) + x[..., 1])
            )
            + 0.059288787
            * numpy.sqrt(
                numpy.square((-0.971160367) + x[..., 0]) + numpy.square((-0.449937439) + x[..., 1])
            )
            + 0.449225381
            * numpy.sqrt(
                numpy.square((-0.706743171) + x[..., 0]) + numpy.square((-0.272020388) + x[..., 1])
            )
            + 0.595490238
            * numpy.sqrt(
                numpy.square((-0.986271722) + x[..., 0]) + numpy.square((-0.055287556) + x[..., 1])
            )
            + 0.445010086
            * numpy.sqrt(
                numpy.square((-0.854820634) + x[..., 0]) + numpy.square((-0.966585121) + x[..., 1])
            )
            + 0.127887704
            * numpy.sqrt(
                numpy.square((-0.62144112) + x[..., 0]) + numpy.square((-0.017750601) + x[..., 1])
            )
            + 0.723567866
            * numpy.sqrt(
                numpy.square((-0.701314879) + x[..., 0]) + numpy.square((-0.017716888) + x[..., 1])
            )
            + 0.72998288
            * numpy.sqrt(
                numpy.square((-0.700888672) + x[..., 0]) + numpy.square((-0.14849054) + x[..., 1])
            )
            + 0.531587725
            * numpy.sqrt(
                numpy.square((-0.790702699) + x[..., 0]) + numpy.square((-0.47073351) + x[..., 1])
            )
            + 0.770201286
            * numpy.sqrt(
                numpy.square((-0.610224526) + x[..., 0]) + numpy.square((-0.724916073) + x[..., 1])
            )
            + 0.443099759
            * numpy.sqrt(
                numpy.square((-0.054312694) + x[..., 0]) + numpy.square((-0.165408535) + x[..., 1])
            )
            + 0.165432987
            * numpy.sqrt(
                numpy.square((-0.485176103) + x[..., 0]) + numpy.square((-0.088575636) + x[..., 1])
            )
            + 0.310543666
            * numpy.sqrt(
                numpy.square((-0.052547941) + x[..., 0]) + numpy.square((-0.264142699) + x[..., 1])
            )
            + 0.020504433
            * numpy.sqrt(
                numpy.square((-0.698580858) + x[..., 0]) + numpy.square((-0.402312087) + x[..., 1])
            )
            + 0.805351979
            * numpy.sqrt(
                numpy.square((-0.194783617) + x[..., 0]) + numpy.square((-0.904466729) + x[..., 1])
            )
            + 0.640101716
            * numpy.sqrt(
                numpy.square((-0.226034356) + x[..., 0]) + numpy.square((-0.448168695) + x[..., 1])
            )
            + 0.34177506
            * numpy.sqrt(
                numpy.square((-0.813635238) + x[..., 0]) + numpy.square((-0.34878602) + x[..., 1])
            )
            + 0.475518068
            * numpy.sqrt(
                numpy.square((-0.991730517) + x[..., 0]) + numpy.square((-0.505610118) + x[..., 1])
            )
            + 0.088685299
            * numpy.sqrt(
                numpy.square((-0.750669929) + x[..., 0]) + numpy.square((-0.395292087) + x[..., 1])
            )
            + 0.972815915
            * numpy.sqrt(
                numpy.square((-0.718343639) + x[..., 0]) + numpy.square((-0.056940821) + x[..., 1])
            )
            + 0.213692245
            * numpy.sqrt(
                numpy.square((-0.000591136) + x[..., 0]) + numpy.square((-0.716713819) + x[..., 1])
            )
            + 0.510888759
            * numpy.sqrt(
                numpy.square((-0.263857554) + x[..., 0]) + numpy.square((-0.179658395) + x[..., 1])
            )
            + 0.627937366
            * numpy.sqrt(
                numpy.square((-0.823820009) + x[..., 0]) + numpy.square((-0.623093467) + x[..., 1])
            )
            + 0.812010818
            * numpy.sqrt(
                numpy.square((-0.819528513) + x[..., 0]) + numpy.square((-0.401520664) + x[..., 1])
            )
            + 0.934184816
            * numpy.sqrt(
                numpy.square((-0.860411595) + x[..., 0]) + numpy.square((-0.117278922) + x[..., 1])
            )
            + 0.567321832
            * numpy.sqrt(
                numpy.square((-0.212686822) + x[..., 0]) + numpy.square((-0.72429774) + x[..., 1])
            )
            + 0.611810573
            * numpy.sqrt(
                numpy.square((-0.456789096) + x[..., 0]) + numpy.square((-0.71572783) + x[..., 1])
            )
            + 0.512077681
            * numpy.sqrt(
                numpy.square((-0.038362715) + x[..., 0]) + numpy.square((-0.074068741) + x[..., 1])
            )
            + 0.05938705
            * numpy.sqrt(
                numpy.square((-0.32300194) + x[..., 0]) + numpy.square((-0.180901788) + x[..., 1])
            )
            + 0.849493574
            * numpy.sqrt(
                numpy.square((-0.439877392) + x[..., 0]) + numpy.square((-0.639837973) + x[..., 1])
            )
            + 0.831244013
            * numpy.sqrt(
                numpy.square((-0.315329019) + x[..., 0]) + numpy.square((-0.145447766) + x[..., 1])
            )
            + 0.961697215
            * numpy.sqrt(
                numpy.square((-0.134766179) + x[..., 0]) + numpy.square((-0.663315308) + x[..., 1])
            )
            + 0.373786877
            * numpy.sqrt(
                numpy.square((-0.810956334) + x[..., 0]) + numpy.square((-0.131925734) + x[..., 1])
            )
            + 0.113648288
            * numpy.sqrt(
                numpy.square((-0.416792254) + x[..., 0]) + numpy.square((-0.70742115) + x[..., 1])
            )
            + 0.792993877
            * numpy.sqrt(
                numpy.square((-0.14178117) + x[..., 0]) + numpy.square((-0.570574529) + x[..., 1])
            )
            + 0.696166945
            * numpy.sqrt(
                numpy.square((-0.465534642) + x[..., 0]) + numpy.square((-0.033405054) + x[..., 1])
            )
            + 0.443765529
            * numpy.sqrt(
                numpy.square((-0.282993813) + x[..., 0]) + numpy.square((-0.285193139) + x[..., 1])
            )
            + 0.179455474
            * numpy.sqrt(
                numpy.square((-0.895681633) + x[..., 0]) + numpy.square((-0.405829907) + x[..., 1])
            )
            + 0.628691798
            * numpy.sqrt(
                numpy.square((-0.064407708) + x[..., 0]) + numpy.square((-0.830836969) + x[..., 1])
            )
            + 0.920973065
            * numpy.sqrt(
                numpy.square((-0.414599358) + x[..., 0]) + numpy.square((-0.094408149) + x[..., 1])
            )
        )
        y6 = (
            0.075975859
            * numpy.sqrt(
                numpy.square((-0.341613792) + x[..., 0]) + numpy.square((-0.155522578) + x[..., 1])
            )
            + 0.716392187
            * numpy.sqrt(
                numpy.square((-0.468286051) + x[..., 0]) + numpy.square((-0.903303755) + x[..., 1])
            )
            + 0.826324989
            * numpy.sqrt(
                numpy.square((-0.642670025) + x[..., 0]) + numpy.square((-0.457772801) + x[..., 1])
            )
            + 0.990758013
            * numpy.sqrt(
                numpy.square((-0.643575647) + x[..., 0]) + numpy.square((-0.896922927) + x[..., 1])
            )
            + 0.505685812
            * numpy.sqrt(
                numpy.square((-0.337605676) + x[..., 0]) + numpy.square((-0.865178568) + x[..., 1])
            )
            + 0.525714161
            * numpy.sqrt(
                numpy.square((-0.100816091) + x[..., 0]) + numpy.square((-0.546208503) + x[..., 1])
            )
            + 0.868866586
            * numpy.sqrt(
                numpy.square((-0.90582698) + x[..., 0]) + numpy.square((-0.321590338) + x[..., 1])
            )
            + 0.268735293
            * numpy.sqrt(numpy.square((-0.2173508) + x[..., 0]) + numpy.square((-0.96335202) + x[..., 1]))
            + 0.603890027
            * numpy.sqrt(
                numpy.square((-0.91886627) + x[..., 0]) + numpy.square((-0.910526384) + x[..., 1])
            )
            + 0.060719731
            * numpy.sqrt(
                numpy.square((-0.451754435) + x[..., 0]) + numpy.square((-0.083007725) + x[..., 1])
            )
            + 0.826749167
            * numpy.sqrt(
                numpy.square((-0.08993069) + x[..., 0]) + numpy.square((-0.638712017) + x[..., 1])
            )
            + 0.745893015
            * numpy.sqrt(
                numpy.square((-0.3741985) + x[..., 0]) + numpy.square((-0.078355642) + x[..., 1])
            )
            + 0.105636691
            * numpy.sqrt(
                numpy.square((-0.414989978) + x[..., 0]) + numpy.square((-0.332566233) + x[..., 1])
            )
            + 0.241659551
            * numpy.sqrt(
                numpy.square((-0.404194814) + x[..., 0]) + numpy.square((-0.103033723) + x[..., 1])
            )
            + 0.701580004
            * numpy.sqrt(
                numpy.square((-0.111668833) + x[..., 0]) + numpy.square((-0.254086121) + x[..., 1])
            )
            + 0.419469585
            * numpy.sqrt(
                numpy.square((-0.751128625) + x[..., 0]) + numpy.square((-0.747202699) + x[..., 1])
            )
            + 0.431247224
            * numpy.sqrt(
                numpy.square((-0.803403987) + x[..., 0]) + numpy.square((-0.0908481) + x[..., 1])
            )
            + 0.010665641
            * numpy.sqrt(
                numpy.square((-0.023657509) + x[..., 0]) + numpy.square((-0.593507058) + x[..., 1])
            )
            + 0.337519542
            * numpy.sqrt(
                numpy.square((-0.480878863) + x[..., 0]) + numpy.square((-0.709875838) + x[..., 1])
            )
            + 0.279986345
            * numpy.sqrt(
                numpy.square((-0.278585554) + x[..., 0]) + numpy.square((-0.399851972) + x[..., 1])
            )
            + 0.036734579
            * numpy.sqrt(
                numpy.square((-0.901613874) + x[..., 0]) + numpy.square((-0.217434149) + x[..., 1])
            )
            + 0.190743849
            * numpy.sqrt(
                numpy.square((-0.017586434) + x[..., 0]) + numpy.square((-0.79524831) + x[..., 1])
            )
            + 0.34881608
            * numpy.sqrt(
                numpy.square((-0.681039283) + x[..., 0]) + numpy.square((-0.721782274) + x[..., 1])
            )
            + 0.726553624
            * numpy.sqrt(
                numpy.square((-0.950906732) + x[..., 0]) + numpy.square((-0.788745215) + x[..., 1])
            )
            + 0.645887357
            * numpy.sqrt(
                numpy.square((-0.900175443) + x[..., 0]) + numpy.square((-0.423987865) + x[..., 1])
            )
            + 0.31258634
            * numpy.sqrt(
                numpy.square((-0.898803246) + x[..., 0]) + numpy.square((-0.884958168) + x[..., 1])
            )
            + 0.189279697
            * numpy.sqrt(
                numpy.square((-0.874462095) + x[..., 0]) + numpy.square((-0.65531861) + x[..., 1])
            )
            + 0.830043558
            * numpy.sqrt(
                numpy.square((-0.390995248) + x[..., 0]) + numpy.square((-0.761515411) + x[..., 1])
            )
            + 0.65030511
            * numpy.sqrt(
                numpy.square((-0.504214104) + x[..., 0]) + numpy.square((-0.20123387) + x[..., 1])
            )
            + 0.827844101
            * numpy.sqrt(
                numpy.square((-0.831264581) + x[..., 0]) + numpy.square((-0.351847486) + x[..., 1])
            )
            + 0.992575027
            * numpy.sqrt(
                numpy.square((-0.602137697) + x[..., 0]) + numpy.square((-0.565015331) + x[..., 1])
            )
            + 0.619911844
            * numpy.sqrt(
                numpy.square((-0.082245972) + x[..., 0]) + numpy.square((-0.769275577) + x[..., 1])
            )
            + 0.446539541
            * numpy.sqrt(
                numpy.square((-0.57775716) + x[..., 0]) + numpy.square((-0.717348195) + x[..., 1])
            )
            + 0.073080613
            * numpy.sqrt(
                numpy.square((-0.593176007) + x[..., 0]) + numpy.square((-0.282200676) + x[..., 1])
            )
            + 0.152014344
            * numpy.sqrt(
                numpy.square((-0.683772744) + x[..., 0]) + numpy.square((-0.398502483) + x[..., 1])
            )
            + 0.057701323
            * numpy.sqrt(
                numpy.square((-0.158771356) + x[..., 0]) + numpy.square((-0.650301381) + x[..., 1])
            )
            + 0.157330497
            * numpy.sqrt(
                numpy.square((-0.331776882) + x[..., 0]) + numpy.square((-0.281602406) + x[..., 1])
            )
            + 0.766495753
            * numpy.sqrt(
                numpy.square((-0.315855332) + x[..., 0]) + numpy.square((-0.736487798) + x[..., 1])
            )
            + 0.08721151
            * numpy.sqrt(
                numpy.square((-0.519931741) + x[..., 0]) + numpy.square((-0.668893067) + x[..., 1])
            )
            + 0.060442143
            * numpy.sqrt(
                numpy.square((-0.363788165) + x[..., 0]) + numpy.square((-0.489404343) + x[..., 1])
            )
            + 0.753357725
            * numpy.sqrt(
                numpy.square((-0.16775638) + x[..., 0]) + numpy.square((-0.359089339) + x[..., 1])
            )
            + 0.946317737
            * numpy.sqrt(
                numpy.square((-0.683085662) + x[..., 0]) + numpy.square((-0.67851333) + x[..., 1])
            )
            + 0.318200177
            * numpy.sqrt(
                numpy.square((-0.505392859) + x[..., 0]) + numpy.square((-0.345411718) + x[..., 1])
            )
            + 0.585366695
            * numpy.sqrt(
                numpy.square((-0.57623508) + x[..., 0]) + numpy.square((-0.566733149) + x[..., 1])
            )
            + 0.50546319
            * numpy.sqrt(
                numpy.square((-0.719827246) + x[..., 0]) + numpy.square((-0.800316452) + x[..., 1])
            )
            + 0.941674827
            * numpy.sqrt(
                numpy.square((-0.683728445) + x[..., 0]) + numpy.square((-0.222468578) + x[..., 1])
            )
            + 0.476412446
            * numpy.sqrt(
                numpy.square((-0.019849389) + x[..., 0]) + numpy.square((-0.426708266) + x[..., 1])
            )
            + 0.509861485
            * numpy.sqrt(
                numpy.square((-0.839795967) + x[..., 0]) + numpy.square((-0.681988295) + x[..., 1])
            )
            + 0.71479322
            * numpy.sqrt(
                numpy.square((-0.710049083) + x[..., 0]) + numpy.square((-0.153715405) + x[..., 1])
            )
            + 0.967552016
            * numpy.sqrt(
                numpy.square((-0.155509448) + x[..., 0]) + numpy.square((-0.127754767) + x[..., 1])
            )
            + 0.47556765
            * numpy.sqrt(
                numpy.square((-0.610714008) + x[..., 0]) + numpy.square((-0.867719998) + x[..., 1])
            )
            + 0.256524524
            * numpy.sqrt(
                numpy.square((-0.661552693) + x[..., 0]) + numpy.square((-0.765041719) + x[..., 1])
            )
            + 0.819955638
            * numpy.sqrt(
                numpy.square((-0.194366754) + x[..., 0]) + numpy.square((-0.578964054) + x[..., 1])
            )
            + 0.241572933
            * numpy.sqrt(
                numpy.square((-0.363519036) + x[..., 0]) + numpy.square((-0.788560262) + x[..., 1])
            )
            + 0.568206318
            * numpy.sqrt(
                numpy.square((-0.623896659) + x[..., 0]) + numpy.square((-0.896803129) + x[..., 1])
            )
            + 0.54373679
            * numpy.sqrt(
                numpy.square((-0.731389266) + x[..., 0]) + numpy.square((-0.438955863) + x[..., 1])
            )
            + 0.196799724
            * numpy.sqrt(
                numpy.square((-0.413973357) + x[..., 0]) + numpy.square((-0.588920116) + x[..., 1])
            )
            + 0.333212237
            * numpy.sqrt(
                numpy.square((-0.157493922) + x[..., 0]) + numpy.square((-0.835859245) + x[..., 1])
            )
            + 0.604479984
            * numpy.sqrt(
                numpy.square((-0.012519218) + x[..., 0]) + numpy.square((-0.420600655) + x[..., 1])
            )
            + 0.034836176
            * numpy.sqrt(
                numpy.square((-0.010171892) + x[..., 0]) + numpy.square((-0.069774605) + x[..., 1])
            )
            + 0.326561858
            * numpy.sqrt(
                numpy.square((-0.952031263) + x[..., 0]) + numpy.square((-0.025197986) + x[..., 1])
            )
            + 0.797264624
            * numpy.sqrt(
                numpy.square((-0.97667921) + x[..., 0]) + numpy.square((-0.850983156) + x[..., 1])
            )
            + 0.724799308
            * numpy.sqrt(
                numpy.square((-0.966318651) + x[..., 0]) + numpy.square((-0.349599021) + x[..., 1])
            )
            + 0.873742673
            * numpy.sqrt(
                numpy.square((-0.856279221) + x[..., 0]) + numpy.square((-0.096481944) + x[..., 1])
            )
            + 0.813964107
            * numpy.sqrt(
                numpy.square((-0.141610255) + x[..., 0]) + numpy.square((-0.25857985) + x[..., 1])
            )
            + 0.665492836
            * numpy.sqrt(
                numpy.square((-0.049733593) + x[..., 0]) + numpy.square((-0.137132461) + x[..., 1])
            )
            + 0.193877504
            * numpy.sqrt(
                numpy.square((-0.55303282) + x[..., 0]) + numpy.square((-0.293232829) + x[..., 1])
            )
            + 0.474020355
            * numpy.sqrt(
                numpy.square((-0.184029187) + x[..., 0]) + numpy.square((-0.59717233) + x[..., 1])
            )
            + 0.344030384
            * numpy.sqrt(
                numpy.square((-0.994166462) + x[..., 0]) + numpy.square((-0.688136263) + x[..., 1])
            )
            + 0.889569186
            * numpy.sqrt(
                numpy.square((-0.809087892) + x[..., 0]) + numpy.square((-0.548545074) + x[..., 1])
            )
            + 0.485284769
            * numpy.sqrt(
                numpy.square((-0.30620673) + x[..., 0]) + numpy.square((-0.16832474) + x[..., 1])
            )
            + 0.863925392
            * numpy.sqrt(
                numpy.square((-0.087402042) + x[..., 0]) + numpy.square((-0.814766486) + x[..., 1])
            )
            + 0.398855068
            * numpy.sqrt(
                numpy.square((-0.430502537) + x[..., 0]) + numpy.square((-0.553705745) + x[..., 1])
            )
            + 0.825441861
            * numpy.sqrt(
                numpy.square((-0.349684504) + x[..., 0]) + numpy.square((-0.442819762) + x[..., 1])
            )
            + 0.220599295
            * numpy.sqrt(
                numpy.square((-0.117340452) + x[..., 0]) + numpy.square((-0.165685056) + x[..., 1])
            )
            + 0.187241005
            * numpy.sqrt(
                numpy.square((-0.585981442) + x[..., 0]) + numpy.square((-0.52015142) + x[..., 1])
            )
            + 0.359499836
            * numpy.sqrt(
                numpy.square((-0.445526822) + x[..., 0]) + numpy.square((-0.826865853) + x[..., 1])
            )
            + 0.387118865
            * numpy.sqrt(
                numpy.square((-0.412318519) + x[..., 0]) + numpy.square((-0.042276707) + x[..., 1])
            )
            + 0.352766747
            * numpy.sqrt(
                numpy.square((-0.914514752) + x[..., 0]) + numpy.square((-0.19760502) + x[..., 1])
            )
            + 0.532239069
            * numpy.sqrt(
                numpy.square((-0.213783869) + x[..., 0]) + numpy.square((-0.716566715) + x[..., 1])
            )
            + 0.519592463
            * numpy.sqrt(
                numpy.square((-0.224172661) + x[..., 0]) + numpy.square((-0.314383639) + x[..., 1])
            )
            + 0.493112752
            * numpy.sqrt(
                numpy.square((-0.542333641) + x[..., 0]) + numpy.square((-0.621913556) + x[..., 1])
            )
            + 0.386278029
            * numpy.sqrt(
                numpy.square((-0.631056429) + x[..., 0]) + numpy.square((-0.476322272) + x[..., 1])
            )
            + 0.470849636
            * numpy.sqrt(
                numpy.square((-0.327433784) + x[..., 0]) + numpy.square((-0.795403963) + x[..., 1])
            )
            + 0.199152303
            * numpy.sqrt(
                numpy.square((-0.148784828) + x[..., 0]) + numpy.square((-0.521010517) + x[..., 1])
            )
            + 0.071601962
            * numpy.sqrt(
                numpy.square((-0.92914777) + x[..., 0]) + numpy.square((-0.470607182) + x[..., 1])
            )
            + 0.543935985
            * numpy.sqrt(
                numpy.square((-0.251032149) + x[..., 0]) + numpy.square((-0.613752999) + x[..., 1])
            )
            + 0.730147354
            * numpy.sqrt(
                numpy.square((-0.062587049) + x[..., 0]) + numpy.square((-0.81404444) + x[..., 1])
            )
            + 0.246755624
            * numpy.sqrt(
                numpy.square((-0.31014418) + x[..., 0]) + numpy.square((-0.824427875) + x[..., 1])
            )
            + 0.161256331
            * numpy.sqrt(
                numpy.square((-0.040197097) + x[..., 0]) + numpy.square((-0.501579556) + x[..., 1])
            )
            + 0.552015511
            * numpy.sqrt(
                numpy.square((-0.82116568) + x[..., 0]) + numpy.square((-0.785122813) + x[..., 1])
            )
            + 0.631616336
            * numpy.sqrt(
                numpy.square((-0.230960791) + x[..., 0]) + numpy.square((-0.735393903) + x[..., 1])
            )
            + 0.897629167
            * numpy.sqrt(
                numpy.square((-0.410028352) + x[..., 0]) + numpy.square((-0.41489746) + x[..., 1])
            )
            + 0.362611397
            * numpy.sqrt(
                numpy.square((-0.302580941) + x[..., 0]) + numpy.square((-0.705541047) + x[..., 1])
            )
            + 0.7200471
            * numpy.sqrt(
                numpy.square((-0.444921895) + x[..., 0]) + numpy.square((-0.578877959) + x[..., 1])
            )
            + 0.852937005
            * numpy.sqrt(
                numpy.square((-0.716001945) + x[..., 0]) + numpy.square((-0.276081614) + x[..., 1])
            )
            + 0.837601248
            * numpy.sqrt(
                numpy.square((-0.593154972) + x[..., 0]) + numpy.square((-0.039801313) + x[..., 1])
            )
            + 0.641502008
            * numpy.sqrt(
                numpy.square((-0.131194359) + x[..., 0]) + numpy.square((-0.267056076) + x[..., 1])
            )
            + 0.416400818
            * numpy.sqrt(
                numpy.square((-0.161245102) + x[..., 0]) + numpy.square((-0.234166916) + x[..., 1])
            )
            + 0.261797349
            * numpy.sqrt(
                numpy.square((-0.315632432) + x[..., 0]) + numpy.square((-0.697282729) + x[..., 1])
            )
            + 0.139317151
            * numpy.sqrt(
                numpy.square((-0.572059612) + x[..., 0]) + numpy.square((-0.968382457) + x[..., 1])
            )
            + 0.05156982
            * numpy.sqrt(
                numpy.square((-0.268720764) + x[..., 0]) + numpy.square((-0.679505567) + x[..., 1])
            )
            + 0.399293818
            * numpy.sqrt(
                numpy.square((-0.03639198) + x[..., 0]) + numpy.square((-0.591140195) + x[..., 1])
            )
            + 0.262255889
            * numpy.sqrt(
                numpy.square((-0.686391574) + x[..., 0]) + numpy.square((-0.502678512) + x[..., 1])
            )
            + 0.460310332
            * numpy.sqrt(
                numpy.square((-0.674630585) + x[..., 0]) + numpy.square((-0.658758252) + x[..., 1])
            )
            + 0.84033505
            * numpy.sqrt(
                numpy.square((-0.332128454) + x[..., 0]) + numpy.square((-0.339029528) + x[..., 1])
            )
            + 0.574746709
            * numpy.sqrt(
                numpy.square((-0.759938819) + x[..., 0]) + numpy.square((-0.254036765) + x[..., 1])
            )
            + 0.402130629
            * numpy.sqrt(
                numpy.square((-0.17678032) + x[..., 0]) + numpy.square((-0.171935584) + x[..., 1])
            )
            + 0.330866514
            * numpy.sqrt(
                numpy.square((-0.682479753) + x[..., 0]) + numpy.square((-0.257464382) + x[..., 1])
            )
            + 0.725301056
            * numpy.sqrt(
                numpy.square((-0.672989927) + x[..., 0]) + numpy.square((-0.475571818) + x[..., 1])
            )
            + 0.726290941
            * numpy.sqrt(
                numpy.square((-0.831213823) + x[..., 0]) + numpy.square((-0.876104553) + x[..., 1])
            )
            + 0.644759277
            * numpy.sqrt(
                numpy.square((-0.515170111) + x[..., 0]) + numpy.square((-0.402291244) + x[..., 1])
            )
        )
        y7 = (
            0.315667135
            * numpy.sqrt(
                numpy.square((-0.283031827) + x[..., 0]) + numpy.square((-0.081665687) + x[..., 1])
            )
            + 0.474726479
            * numpy.sqrt(
                numpy.square((-0.555420166) + x[..., 0]) + numpy.square((-0.709320102) + x[..., 1])
            )
            + 0.383942178
            * numpy.sqrt(
                numpy.square((-0.413991606) + x[..., 0]) + numpy.square((-0.853334696) + x[..., 1])
            )
            + 0.368982488
            * numpy.sqrt(
                numpy.square((-0.073407669) + x[..., 0]) + numpy.square((-0.292747731) + x[..., 1])
            )
            + 0.29386898
            * numpy.sqrt(
                numpy.square((-0.806006615) + x[..., 0]) + numpy.square((-0.277179103) + x[..., 1])
            )
            + 0.724600113
            * numpy.sqrt(
                numpy.square((-0.332716456) + x[..., 0]) + numpy.square((-0.174801885) + x[..., 1])
            )
            + 0.340395043
            * numpy.sqrt(
                numpy.square((-0.084689622) + x[..., 0]) + numpy.square((-0.87135762) + x[..., 1])
            )
            + 0.90042072
            * numpy.sqrt(
                numpy.square((-0.572164261) + x[..., 0]) + numpy.square((-0.27001122) + x[..., 1])
            )
            + 0.108716489
            * numpy.sqrt(
                numpy.square((-0.022055714) + x[..., 0]) + numpy.square((-0.05615356) + x[..., 1])
            )
            + 0.676698169
            * numpy.sqrt(
                numpy.square((-0.742039032) + x[..., 0]) + numpy.square((-0.577884621) + x[..., 1])
            )
            + 0.168965877
            * numpy.sqrt(
                numpy.square((-0.905099659) + x[..., 0]) + numpy.square((-0.886976413) + x[..., 1])
            )
            + 0.243665841
            * numpy.sqrt(
                numpy.square((-0.56081732) + x[..., 0]) + numpy.square((-0.124221959) + x[..., 1])
            )
            + 0.802357676
            * numpy.sqrt(
                numpy.square((-0.472825602) + x[..., 0]) + numpy.square((-0.192603048) + x[..., 1])
            )
            + 0.424842626
            * numpy.sqrt(
                numpy.square((-0.717564056) + x[..., 0]) + numpy.square((-0.685928725) + x[..., 1])
            )
            + 0.110567667
            * numpy.sqrt(
                numpy.square((-0.513010352) + x[..., 0]) + numpy.square((-0.797264722) + x[..., 1])
            )
            + 0.452537331
            * numpy.sqrt(
                numpy.square((-0.887081158) + x[..., 0]) + numpy.square((-0.651036893) + x[..., 1])
            )
            + 0.507241684
            * numpy.sqrt(
                numpy.square((-0.771522965) + x[..., 0]) + numpy.square((-0.468996515) + x[..., 1])
            )
            + 0.6352326
            * numpy.sqrt(
                numpy.square((-0.140124537) + x[..., 0]) + numpy.square((-0.52869047) + x[..., 1])
            )
            + 0.681777815
            * numpy.sqrt(
                numpy.square((-0.264515472) + x[..., 0]) + numpy.square((-0.511004954) + x[..., 1])
            )
            + 0.417825009
            * numpy.sqrt(
                numpy.square((-0.682555102) + x[..., 0]) + numpy.square((-0.40702847) + x[..., 1])
            )
            + 0.910706419
            * numpy.sqrt(
                numpy.square((-0.449804485) + x[..., 0]) + numpy.square((-0.53109588) + x[..., 1])
            )
            + 0.842905262
            * numpy.sqrt(
                numpy.square((-0.965524814) + x[..., 0]) + numpy.square((-0.421231311) + x[..., 1])
            )
            + 0.817445849
            * numpy.sqrt(
                numpy.square((-0.957894832) + x[..., 0]) + numpy.square((-0.56285131) + x[..., 1])
            )
            + 0.552040447
            * numpy.sqrt(
                numpy.square((-0.89922658) + x[..., 0]) + numpy.square((-0.186628872) + x[..., 1])
            )
            + 0.017545102
            * numpy.sqrt(
                numpy.square((-0.327545639) + x[..., 0]) + numpy.square((-0.754193558) + x[..., 1])
            )
            + 0.129753505
            * numpy.sqrt(
                numpy.square((-0.457099052) + x[..., 0]) + numpy.square((-0.711280553) + x[..., 1])
            )
            + 0.650893998
            * numpy.sqrt(
                numpy.square((-0.596180286) + x[..., 0]) + numpy.square((-0.216439909) + x[..., 1])
            )
            + 0.95757937
            * numpy.sqrt(
                numpy.square((-0.878623594) + x[..., 0]) + numpy.square((-0.732450255) + x[..., 1])
            )
            + 0.503395644
            * numpy.sqrt(
                numpy.square((-0.170672595) + x[..., 0]) + numpy.square((-0.573804053) + x[..., 1])
            )
            + 0.000199195
            * numpy.sqrt(
                numpy.square((-0.633602195) + x[..., 0]) + numpy.square((-0.174797136) + x[..., 1])
            )
            + 0.53334763
            * numpy.sqrt(
                numpy.square((-0.771589599) + x[..., 0]) + numpy.square((-0.225124324) + x[..., 1])
            )
            + 0.913543387
            * numpy.sqrt(
                numpy.square((-0.569445994) + x[..., 0]) + numpy.square((-0.98856863) + x[..., 1])
            )
            + 0.556776347
            * numpy.sqrt(
                numpy.square((-0.027677879) + x[..., 0]) + numpy.square((-0.080978901) + x[..., 1])
            )
            + 0.517179335
            * numpy.sqrt(
                numpy.square((-0.810993788) + x[..., 0]) + numpy.square((-0.715348208) + x[..., 1])
            )
            + 0.305054478
            * numpy.sqrt(
                numpy.square((-0.278929528) + x[..., 0]) + numpy.square((-0.710195917) + x[..., 1])
            )
            + 0.100364543
            * numpy.sqrt(
                numpy.square((-0.433349142) + x[..., 0]) + numpy.square((-0.563914304) + x[..., 1])
            )
            + 0.911919538
            * numpy.sqrt(
                numpy.square((-0.33626229) + x[..., 0]) + numpy.square((-0.355942026) + x[..., 1])
            )
            + 0.26824041
            * numpy.sqrt(
                numpy.square((-0.588642674) + x[..., 0]) + numpy.square((-0.482396015) + x[..., 1])
            )
            + 0.550392537
            * numpy.sqrt(
                numpy.square((-0.57439169) + x[..., 0]) + numpy.square((-0.017501764) + x[..., 1])
            )
            + 0.810934627
            * numpy.sqrt(
                numpy.square((-0.543421379) + x[..., 0]) + numpy.square((-0.902668852) + x[..., 1])
            )
            + 0.098040996
            * numpy.sqrt(
                numpy.square((-0.578161539) + x[..., 0]) + numpy.square((-0.973823247) + x[..., 1])
            )
            + 0.183506931
            * numpy.sqrt(
                numpy.square((-0.977215915) + x[..., 0]) + numpy.square((-0.636994586) + x[..., 1])
            )
            + 0.824627071
            * numpy.sqrt(
                numpy.square((-0.32146597) + x[..., 0]) + numpy.square((-0.009146466) + x[..., 1])
            )
            + 0.699570872
            * numpy.sqrt(
                numpy.square((-0.76297172) + x[..., 0]) + numpy.square((-0.419837383) + x[..., 1])
            )
            + 0.346774294
            * numpy.sqrt(
                numpy.square((-0.962514034) + x[..., 0]) + numpy.square((-0.511180827) + x[..., 1])
            )
            + 0.141663156
            * numpy.sqrt(
                numpy.square((-0.948989938) + x[..., 0]) + numpy.square((-0.776373709) + x[..., 1])
            )
            + 0.647319806
            * numpy.sqrt(
                numpy.square((-0.255889037) + x[..., 0]) + numpy.square((-0.698156911) + x[..., 1])
            )
            + 0.594570656
            * numpy.sqrt(
                numpy.square((-0.324946081) + x[..., 0]) + numpy.square((-0.585700363) + x[..., 1])
            )
            + 0.621503268
            * numpy.sqrt(
                numpy.square((-0.214788002) + x[..., 0]) + numpy.square((-0.560357231) + x[..., 1])
            )
            + 0.154766243
            * numpy.sqrt(
                numpy.square((-0.173957377) + x[..., 0]) + numpy.square((-0.627956318) + x[..., 1])
            )
            + 0.123622771
            * numpy.sqrt(
                numpy.square((-0.731253498) + x[..., 0]) + numpy.square((-0.992141449) + x[..., 1])
            )
            + 0.471729915
            * numpy.sqrt(
                numpy.square((-0.270161234) + x[..., 0]) + numpy.square((-0.551620604) + x[..., 1])
            )
            + 0.281689612
            * numpy.sqrt(
                numpy.square((-0.758475175) + x[..., 0]) + numpy.square((-0.033449494) + x[..., 1])
            )
            + 0.171100981
            * numpy.sqrt(
                numpy.square((-0.617429954) + x[..., 0]) + numpy.square((-0.815931755) + x[..., 1])
            )
            + 0.553029162
            * numpy.sqrt(
                numpy.square((-0.29099745) + x[..., 0]) + numpy.square((-0.874420624) + x[..., 1])
            )
            + 0.138457083
            * numpy.sqrt(
                numpy.square((-0.740698186) + x[..., 0]) + numpy.square((-0.666501657) + x[..., 1])
            )
            + 0.919719683
            * numpy.sqrt(
                numpy.square((-0.007763019) + x[..., 0]) + numpy.square((-0.252076314) + x[..., 1])
            )
            + 0.135178809
            * numpy.sqrt(
                numpy.square((-0.866509716) + x[..., 0]) + numpy.square((-0.808830371) + x[..., 1])
            )
            + 0.742269632
            * numpy.sqrt(
                numpy.square((-0.015140824) + x[..., 0]) + numpy.square((-0.817761858) + x[..., 1])
            )
            + 0.515774392
            * numpy.sqrt(
                numpy.square((-0.428284381) + x[..., 0]) + numpy.square((-0.178946868) + x[..., 1])
            )
            + 0.450691859
            * numpy.sqrt(
                numpy.square((-0.358649383) + x[..., 0]) + numpy.square((-0.876109302) + x[..., 1])
            )
            + 0.45180669
            * numpy.sqrt(
                numpy.square((-0.704870893) + x[..., 0]) + numpy.square((-0.04852454) + x[..., 1])
            )
            + 0.302544468
            * numpy.sqrt(
                numpy.square((-0.415870814) + x[..., 0]) + numpy.square((-0.363108277) + x[..., 1])
            )
            + 0.026666621
            * numpy.sqrt(
                numpy.square((-0.549798001) + x[..., 0]) + numpy.square((-0.684494761) + x[..., 1])
            )
            + 0.543461012
            * numpy.sqrt(
                numpy.square((-0.34503685) + x[..., 0]) + numpy.square((-0.715871109) + x[..., 1])
            )
            + 0.232893887
            * numpy.sqrt(
                numpy.square((-0.6995778) + x[..., 0]) + numpy.square((-0.469528227) + x[..., 1])
            )
            + 0.437170278
            * numpy.sqrt(
                numpy.square((-0.933474793) + x[..., 0]) + numpy.square((-0.837486758) + x[..., 1])
            )
            + 0.439746279
            * numpy.sqrt(
                numpy.square((-0.469279768) + x[..., 0]) + numpy.square((-0.011462907) + x[..., 1])
            )
            + 0.704795526
            * numpy.sqrt(
                numpy.square((-0.213611004) + x[..., 0]) + numpy.square((-0.07489033) + x[..., 1])
            )
            + 0.257630662
            * numpy.sqrt(
                numpy.square((-0.510782923) + x[..., 0]) + numpy.square((-0.049774178) + x[..., 1])
            )
            + 0.614936083
            * numpy.sqrt(
                numpy.square((-0.365715493) + x[..., 0]) + numpy.square((-0.804521601) + x[..., 1])
            )
            + 0.865739676
            * numpy.sqrt(
                numpy.square((-0.935400458) + x[..., 0]) + numpy.square((-0.073057889) + x[..., 1])
            )
            + 0.218831782
            * numpy.sqrt(
                numpy.square((-0.068008281) + x[..., 0]) + numpy.square((-0.778672297) + x[..., 1])
            )
            + 0.419980466
            * numpy.sqrt(
                numpy.square((-0.503866822) + x[..., 0]) + numpy.square((-0.626080447) + x[..., 1])
            )
            + 0.278507858
            * numpy.sqrt(
                numpy.square((-0.392408984) + x[..., 0]) + numpy.square((-0.179794135) + x[..., 1])
            )
            + 0.859136599
            * numpy.sqrt(
                numpy.square((-0.204854507) + x[..., 0]) + numpy.square((-0.705843778) + x[..., 1])
            )
            + 0.443611444
            * numpy.sqrt(
                numpy.square((-0.529545228) + x[..., 0]) + numpy.square((-0.442122324) + x[..., 1])
            )
            + 0.55518996
            * numpy.sqrt(
                numpy.square((-0.589086953) + x[..., 0]) + numpy.square((-0.060172798) + x[..., 1])
            )
            + 0.091730115
            * numpy.sqrt(
                numpy.square((-0.345803464) + x[..., 0]) + numpy.square((-0.52787062) + x[..., 1])
            )
            + 0.177210052
            * numpy.sqrt(
                numpy.square((-0.252882174) + x[..., 0]) + numpy.square((-0.41516403) + x[..., 1])
            )
            + 0.555006796
            * numpy.sqrt(
                numpy.square((-0.547657147) + x[..., 0]) + numpy.square((-0.457243788) + x[..., 1])
            )
            + 0.248763369
            * numpy.sqrt(
                numpy.square((-0.54748189) + x[..., 0]) + numpy.square((-0.044504325) + x[..., 1])
            )
            + 0.626712856
            * numpy.sqrt(
                numpy.square((-0.058266845) + x[..., 0]) + numpy.square((-0.474985873) + x[..., 1])
            )
            + 0.778094588
            * numpy.sqrt(
                numpy.square((-0.377722234) + x[..., 0]) + numpy.square((-0.098232235) + x[..., 1])
            )
            + 0.273908254
            * numpy.sqrt(
                numpy.square((-0.974067073) + x[..., 0]) + numpy.square((-0.298692583) + x[..., 1])
            )
            + 0.888588353
            * numpy.sqrt(
                numpy.square((-0.379818729) + x[..., 0]) + numpy.square((-0.82283308) + x[..., 1])
            )
            + 0.597876252
            * numpy.sqrt(
                numpy.square((-0.156293447) + x[..., 0]) + numpy.square((-0.906902943) + x[..., 1])
            )
            + 0.082049868
            * numpy.sqrt(
                numpy.square((-0.472257713) + x[..., 0]) + numpy.square((-0.371658799) + x[..., 1])
            )
            + 0.133237157
            * numpy.sqrt(
                numpy.square((-0.397002182) + x[..., 0]) + numpy.square((-0.862013512) + x[..., 1])
            )
            + 0.93607199
            * numpy.sqrt(
                numpy.square((-0.205521859) + x[..., 0]) + numpy.square((-0.417448186) + x[..., 1])
            )
            + 0.806495563
            * numpy.sqrt(
                numpy.square((-0.627342527) + x[..., 0]) + numpy.square((-0.286735201) + x[..., 1])
            )
            + 0.564034952
            * numpy.sqrt(
                numpy.square((-0.003545834) + x[..., 0]) + numpy.square((-0.181140141) + x[..., 1])
            )
            + 0.151992911
            * numpy.sqrt(
                numpy.square((-0.503953052) + x[..., 0]) + numpy.square((-0.611038395) + x[..., 1])
            )
            + 0.707267238
            * numpy.sqrt(
                numpy.square((-0.002227429) + x[..., 0]) + numpy.square((-0.747490544) + x[..., 1])
            )
            + 0.946598191
            * numpy.sqrt(
                numpy.square((-0.521365665) + x[..., 0]) + numpy.square((-0.846515292) + x[..., 1])
            )
            + 0.552638332
            * numpy.sqrt(
                numpy.square((-0.836122507) + x[..., 0]) + numpy.square((-0.395938626) + x[..., 1])
            )
            + 0.144415402
            * numpy.sqrt(
                numpy.square((-0.072116256) + x[..., 0]) + numpy.square((-0.750018173) + x[..., 1])
            )
            + 0.678177964
            * numpy.sqrt(
                numpy.square((-0.76064865) + x[..., 0]) + numpy.square((-0.884924507) + x[..., 1])
            )
            + 0.171744131
            * numpy.sqrt(
                numpy.square((-0.290146118) + x[..., 0]) + numpy.square((-0.227234335) + x[..., 1])
            )
            + 0.910219178
            * numpy.sqrt(
                numpy.square((-0.244949978) + x[..., 0]) + numpy.square((-0.825252102) + x[..., 1])
            )
            + 0.434425031
            * numpy.sqrt(
                numpy.square((-0.435979586) + x[..., 0]) + numpy.square((-0.979108987) + x[..., 1])
            )
            + 0.84331835
            * numpy.sqrt(
                numpy.square((-0.368807879) + x[..., 0]) + numpy.square((-0.055655841) + x[..., 1])
            )
            + 0.826871976
            * numpy.sqrt(
                numpy.square((-0.553423116) + x[..., 0]) + numpy.square((-0.192709991) + x[..., 1])
            )
            + 0.586178417
            * numpy.sqrt(
                numpy.square((-0.074663465) + x[..., 0]) + numpy.square((-0.06217488) + x[..., 1])
            )
            + 0.559412848
            * numpy.sqrt(
                numpy.square((-0.909441905) + x[..., 0]) + numpy.square((-0.04931362) + x[..., 1])
            )
            + 0.544685785
            * numpy.sqrt(
                numpy.square((-0.048698396) + x[..., 0]) + numpy.square((-0.028118742) + x[..., 1])
            )
            + 0.95936761
            * numpy.sqrt(
                numpy.square((-0.820356749) + x[..., 0]) + numpy.square((-0.596138598) + x[..., 1])
            )
            + 0.979741324
            * numpy.sqrt(
                numpy.square((-0.792950361) + x[..., 0]) + numpy.square((-0.520661979) + x[..., 1])
            )
            + 0.263230827
            * numpy.sqrt(
                numpy.square((-0.659535122) + x[..., 0]) + numpy.square((-0.453915868) + x[..., 1])
            )
            + 0.055869703
            * numpy.sqrt(
                numpy.square((-0.39177087) + x[..., 0]) + numpy.square((-0.932250367) + x[..., 1])
            )
            + 0.582750299
            * numpy.sqrt(
                numpy.square((-0.413152547) + x[..., 0]) + numpy.square((-0.811694858) + x[..., 1])
            )
            + 0.51947913
            * numpy.sqrt(
                numpy.square((-0.865680955) + x[..., 0]) + numpy.square((-0.422788844) + x[..., 1])
            )
        )
        y8 = (
            0.010038211
            * numpy.sqrt(
                numpy.square((-0.975317238) + x[..., 0]) + numpy.square((-0.756467495) + x[..., 1])
            )
            + 0.587286691
            * numpy.sqrt(
                numpy.square((-0.572376221) + x[..., 0]) + numpy.square((-0.894396589) + x[..., 1])
            )
            + 0.019369416
            * numpy.sqrt(
                numpy.square((-0.31396728) + x[..., 0]) + numpy.square((-0.114164127) + x[..., 1])
            )
            + 0.347884586
            * numpy.sqrt(
                numpy.square((-0.455030479) + x[..., 0]) + numpy.square((-0.393348474) + x[..., 1])
            )
            + 0.116247499
            * numpy.sqrt(
                numpy.square((-0.37101957) + x[..., 0]) + numpy.square((-0.100112084) + x[..., 1])
            )
            + 0.843125299
            * numpy.sqrt(
                numpy.square((-0.41984774) + x[..., 0]) + numpy.square((-0.122706991) + x[..., 1])
            )
            + 0.864336436
            * numpy.sqrt(
                numpy.square((-0.085303307) + x[..., 0]) + numpy.square((-0.564912276) + x[..., 1])
            )
            + 0.545402664
            * numpy.sqrt(
                numpy.square((-0.814544384) + x[..., 0]) + numpy.square((-0.662687968) + x[..., 1])
            )
            + 0.039091529
            * numpy.sqrt(
                numpy.square((-0.509196765) + x[..., 0]) + numpy.square((-0.869841899) + x[..., 1])
            )
            + 0.146449107
            * numpy.sqrt(
                numpy.square((-0.734416647) + x[..., 0]) + numpy.square((-0.729698425) + x[..., 1])
            )
            + 0.527826741
            * numpy.sqrt(
                numpy.square((-0.824383317) + x[..., 0]) + numpy.square((-0.706432068) + x[..., 1])
            )
            + 0.436555116
            * numpy.sqrt(
                numpy.square((-0.414528845) + x[..., 0]) + numpy.square((-0.512376475) + x[..., 1])
            )
            + 0.707238125
            * numpy.sqrt(
                numpy.square((-0.924426308) + x[..., 0]) + numpy.square((-0.665391911) + x[..., 1])
            )
            + 0.804001456
            * numpy.sqrt(
                numpy.square((-0.394147829) + x[..., 0]) + numpy.square((-0.64250107) + x[..., 1])
            )
            + 0.76769868
            * numpy.sqrt(
                numpy.square((-0.444314801) + x[..., 0]) + numpy.square((-0.392990372) + x[..., 1])
            )
            + 0.035324851
            * numpy.sqrt(
                numpy.square((-0.695469825) + x[..., 0]) + numpy.square((-0.498182351) + x[..., 1])
            )
            + 0.595353395
            * numpy.sqrt(
                numpy.square((-0.676692965) + x[..., 0]) + numpy.square((-0.578642698) + x[..., 1])
            )
            + 0.578936986
            * numpy.sqrt(
                numpy.square((-0.571542634) + x[..., 0]) + numpy.square((-0.964022707) + x[..., 1])
            )
            + 0.512362305
            * numpy.sqrt(
                numpy.square((-0.173515533) + x[..., 0]) + numpy.square((-0.955482006) + x[..., 1])
            )
            + 0.301948288
            * numpy.sqrt(
                numpy.square((-0.604348565) + x[..., 0]) + numpy.square((-0.512787515) + x[..., 1])
            )
            + 0.907870154
            * numpy.sqrt(
                numpy.square((-0.585994567) + x[..., 0]) + numpy.square((-0.362015887) + x[..., 1])
            )
            + 0.497159459
            * numpy.sqrt(
                numpy.square((-0.727780806) + x[..., 0]) + numpy.square((-0.942170257) + x[..., 1])
            )
            + 0.555706784
            * numpy.sqrt(
                numpy.square((-0.24622699) + x[..., 0]) + numpy.square((-0.854806799) + x[..., 1])
            )
            + 0.400240553
            * numpy.sqrt(
                numpy.square((-0.142082064) + x[..., 0]) + numpy.square((-0.82928747) + x[..., 1])
            )
            + 0.125140598
            * numpy.sqrt(
                numpy.square((-0.891192478) + x[..., 0]) + numpy.square((-0.052362876) + x[..., 1])
            )
            + 0.154982941
            * numpy.sqrt(
                numpy.square((-0.442764603) + x[..., 0]) + numpy.square((-0.923365269) + x[..., 1])
            )
            + 0.496404976
            * numpy.sqrt(
                numpy.square((-0.114317101) + x[..., 0]) + numpy.square((-0.064782741) + x[..., 1])
            )
            + 0.102807273
            * numpy.sqrt(
                numpy.square((-0.903618904) + x[..., 0]) + numpy.square((-0.482760828) + x[..., 1])
            )
            + 0.335559191
            * numpy.sqrt(
                numpy.square((-0.333851323) + x[..., 0]) + numpy.square((-0.948412456) + x[..., 1])
            )
            + 0.459419169
            * numpy.sqrt(
                numpy.square((-0.996023074) + x[..., 0]) + numpy.square((-0.240401286) + x[..., 1])
            )
            + 0.162330185
            * numpy.sqrt(
                numpy.square((-0.464494694) + x[..., 0]) + numpy.square((-0.119582485) + x[..., 1])
            )
            + 0.998058348
            * numpy.sqrt(
                numpy.square((-0.530492466) + x[..., 0]) + numpy.square((-0.053183141) + x[..., 1])
            )
            + 0.193802358
            * numpy.sqrt(
                numpy.square((-0.190381035) + x[..., 0]) + numpy.square((-0.599686328) + x[..., 1])
            )
            + 0.290721171
            * numpy.sqrt(
                numpy.square((-0.199058146) + x[..., 0]) + numpy.square((-0.107788333) + x[..., 1])
            )
            + 0.055726812
            * numpy.sqrt(
                numpy.square((-0.644896451) + x[..., 0]) + numpy.square((-0.305030839) + x[..., 1])
            )
            + 0.3154349
            * numpy.sqrt(
                numpy.square((-0.799082159) + x[..., 0]) + numpy.square((-0.562513855) + x[..., 1])
            )
            + 0.740810809
            * numpy.sqrt(
                numpy.square((-0.586356615) + x[..., 0]) + numpy.square((-0.384382267) + x[..., 1])
            )
            + 0.72511281
            * numpy.sqrt(
                numpy.square((-0.971567664) + x[..., 0]) + numpy.square((-0.162020531) + x[..., 1])
            )
            + 0.995255805
            * numpy.sqrt(
                numpy.square((-0.491085657) + x[..., 0]) + numpy.square((-0.680067517) + x[..., 1])
            )
            + 0.71183954
            * numpy.sqrt(
                numpy.square((-0.372538456) + x[..., 0]) + numpy.square((-0.280489946) + x[..., 1])
            )
            + 0.803080612
            * numpy.sqrt(
                numpy.square((-0.827173857) + x[..., 0]) + numpy.square((-0.047437749) + x[..., 1])
            )
            + 0.207952342
            * numpy.sqrt(
                numpy.square((-0.82086635) + x[..., 0]) + numpy.square((-0.215771428) + x[..., 1])
            )
            + 0.625656973
            * numpy.sqrt(
                numpy.square((-0.031338974) + x[..., 0]) + numpy.square((-0.750361772) + x[..., 1])
            )
            + 0.434454144
            * numpy.sqrt(
                numpy.square((-0.925196663) + x[..., 0]) + numpy.square((-0.929334809) + x[..., 1])
            )
            + 0.985915085
            * numpy.sqrt(
                numpy.square((-0.003092386) + x[..., 0]) + numpy.square((-0.010263026) + x[..., 1])
            )
            + 0.611811628
            * numpy.sqrt(
                numpy.square((-0.618022658) + x[..., 0]) + numpy.square((-0.817099168) + x[..., 1])
            )
            + 0.695268968
            * numpy.sqrt(
                numpy.square((-0.006655184) + x[..., 0]) + numpy.square((-0.974640576) + x[..., 1])
            )
            + 0.642237417
            * numpy.sqrt(
                numpy.square((-0.405575083) + x[..., 0]) + numpy.square((-0.788410246) + x[..., 1])
            )
            + 0.13749293
            * numpy.sqrt(
                numpy.square((-0.656289412) + x[..., 0]) + numpy.square((-0.573435395) + x[..., 1])
            )
            + 0.357224483
            * numpy.sqrt(
                numpy.square((-0.615502242) + x[..., 0]) + numpy.square((-0.878641889) + x[..., 1])
            )
            + 0.112218067
            * numpy.sqrt(
                numpy.square((-0.263405133) + x[..., 0]) + numpy.square((-0.271817362) + x[..., 1])
            )
            + 0.198679023
            * numpy.sqrt(
                numpy.square((-0.070448169) + x[..., 0]) + numpy.square((-0.147555822) + x[..., 1])
            )
            + 0.38558222
            * numpy.sqrt(
                numpy.square((-0.045967406) + x[..., 0]) + numpy.square((-0.182790101) + x[..., 1])
            )
            + 0.613221932
            * numpy.sqrt(
                numpy.square((-0.160270373) + x[..., 0]) + numpy.square((-0.019062939) + x[..., 1])
            )
            + 0.678651425
            * numpy.sqrt(
                numpy.square((-0.452481009) + x[..., 0]) + numpy.square((-0.642814994) + x[..., 1])
            )
            + 0.429583398
            * numpy.sqrt(
                numpy.square((-0.168871538) + x[..., 0]) + numpy.square((-0.732223361) + x[..., 1])
            )
            + 0.39167136
            * numpy.sqrt(
                numpy.square((-0.571598327) + x[..., 0]) + numpy.square((-0.567169918) + x[..., 1])
            )
            + 0.502705764
            * numpy.sqrt(
                numpy.square((-0.858706747) + x[..., 0]) + numpy.square((-0.570043365) + x[..., 1])
            )
            + 0.50830814
            * numpy.sqrt(
                numpy.square((-0.035865177) + x[..., 0]) + numpy.square((-0.364503514) + x[..., 1])
            )
            + 0.836558011
            * numpy.sqrt(
                numpy.square((-0.355354847) + x[..., 0]) + numpy.square((-0.083949995) + x[..., 1])
            )
            + 0.966456429
            * numpy.sqrt(
                numpy.square((-0.337868049) + x[..., 0]) + numpy.square((-0.694000563) + x[..., 1])
            )
            + 0.221485381
            * numpy.sqrt(
                numpy.square((-0.486489568) + x[..., 0]) + numpy.square((-0.868118635) + x[..., 1])
            )
            + 0.557382527
            * numpy.sqrt(
                numpy.square((-0.259369156) + x[..., 0]) + numpy.square((-0.365972322) + x[..., 1])
            )
            + 0.432575862
            * numpy.sqrt(
                numpy.square((-0.891191951) + x[..., 0]) + numpy.square((-0.16455216) + x[..., 1])
            )
            + 0.875097352
            * numpy.sqrt(
                numpy.square((-0.849325781) + x[..., 0]) + numpy.square((-0.736074219) + x[..., 1])
            )
            + 0.963791789
            * numpy.sqrt(
                numpy.square((-0.634517345) + x[..., 0]) + numpy.square((-0.794749703) + x[..., 1])
            )
            + 0.284562205
            * numpy.sqrt(
                numpy.square((-0.986466256) + x[..., 0]) + numpy.square((-0.049974704) + x[..., 1])
            )
            + 0.999970887
            * numpy.sqrt(
                numpy.square((-0.757917433) + x[..., 0]) + numpy.square((-0.403716635) + x[..., 1])
            )
            + 0.857403265
            * numpy.sqrt(
                numpy.square((-0.507883472) + x[..., 0]) + numpy.square((-0.045392815) + x[..., 1])
            )
            + 0.215060348
            * numpy.sqrt(
                numpy.square((-0.767773528) + x[..., 0]) + numpy.square((-0.375610823) + x[..., 1])
            )
            + 0.890909449
            * numpy.sqrt(
                numpy.square((-0.832128134) + x[..., 0]) + numpy.square((-0.087534304) + x[..., 1])
            )
            + 0.917175431
            * numpy.sqrt(
                numpy.square((-0.583935755) + x[..., 0]) + numpy.square((-0.260903374) + x[..., 1])
            )
            + 0.407192855
            * numpy.sqrt(
                numpy.square((-0.575059601) + x[..., 0]) + numpy.square((-0.454683347) + x[..., 1])
            )
            + 0.602143127
            * numpy.sqrt(
                numpy.square((-0.556705404) + x[..., 0]) + numpy.square((-0.717496709) + x[..., 1])
            )
            + 0.867523257
            * numpy.sqrt(
                numpy.square((-0.60357821) + x[..., 0]) + numpy.square((-0.248844617) + x[..., 1])
            )
            + 0.064551804
            * numpy.sqrt(
                numpy.square((-0.97705001) + x[..., 0]) + numpy.square((-0.306360046) + x[..., 1])
            )
            + 0.670287483
            * numpy.sqrt(
                numpy.square((-0.554072949) + x[..., 0]) + numpy.square((-0.749460266) + x[..., 1])
            )
            + 0.969528367
            * numpy.sqrt(
                numpy.square((-0.935041631) + x[..., 0]) + numpy.square((-0.792631919) + x[..., 1])
            )
            + 0.840827705
            * numpy.sqrt(
                numpy.square((-0.413199946) + x[..., 0]) + numpy.square((-0.77997385) + x[..., 1])
            )
            + 0.580454813
            * numpy.sqrt(
                numpy.square((-0.8064457) + x[..., 0]) + numpy.square((-0.024244134) + x[..., 1])
            )
            + 0.195615331
            * numpy.sqrt(
                numpy.square((-0.000777894) + x[..., 0]) + numpy.square((-0.327226671) + x[..., 1])
            )
            + 0.516663652
            * numpy.sqrt(
                numpy.square((-0.455260533) + x[..., 0]) + numpy.square((-0.544120762) + x[..., 1])
            )
            + 0.839576446
            * numpy.sqrt(
                numpy.square((-0.419165302) + x[..., 0]) + numpy.square((-0.02884496) + x[..., 1])
            )
            + 0.279689488
            * numpy.sqrt(
                numpy.square((-0.015664723) + x[..., 0]) + numpy.square((-0.016162089) + x[..., 1])
            )
            + 0.87666887
            * numpy.sqrt(
                numpy.square((-0.081979691) + x[..., 0]) + numpy.square((-0.428706428) + x[..., 1])
            )
            + 0.642851055
            * numpy.sqrt(
                numpy.square((-0.598813739) + x[..., 0]) + numpy.square((-0.696793641) + x[..., 1])
            )
            + 0.988020137
            * numpy.sqrt(
                numpy.square((-0.555175228) + x[..., 0]) + numpy.square((-0.296715646) + x[..., 1])
            )
            + 0.606515667
            * numpy.sqrt(
                numpy.square((-0.618004814) + x[..., 0]) + numpy.square((-0.705289739) + x[..., 1])
            )
            + 0.271351755
            * numpy.sqrt(
                numpy.square((-0.885090866) + x[..., 0]) + numpy.square((-0.993624206) + x[..., 1])
            )
            + 0.188316095
            * numpy.sqrt(
                numpy.square((-0.189865972) + x[..., 0]) + numpy.square((-0.911682365) + x[..., 1])
            )
            + 0.372984157
            * numpy.sqrt(
                numpy.square((-0.428062589) + x[..., 0]) + numpy.square((-0.462401771) + x[..., 1])
            )
            + 0.037801318
            * numpy.sqrt(
                numpy.square((-0.166508875) + x[..., 0]) + numpy.square((-0.261675276) + x[..., 1])
            )
            + 0.467827059
            * numpy.sqrt(
                numpy.square((-0.886264357) + x[..., 0]) + numpy.square((-0.597108255) + x[..., 1])
            )
            + 0.996310017
            * numpy.sqrt(
                numpy.square((-0.676541273) + x[..., 0]) + numpy.square((-0.017379549) + x[..., 1])
            )
            + 0.494107985
            * numpy.sqrt(
                numpy.square((-0.863341691) + x[..., 0]) + numpy.square((-0.410648047) + x[..., 1])
            )
            + 0.696629251
            * numpy.sqrt(
                numpy.square((-0.09275721) + x[..., 0]) + numpy.square((-0.317739324) + x[..., 1])
            )
            + 0.299887365
            * numpy.sqrt(
                numpy.square((-0.996483033) + x[..., 0]) + numpy.square((-0.50933936) + x[..., 1])
            )
            + 0.656529668
            * numpy.sqrt(
                numpy.square((-0.616810129) + x[..., 0]) + numpy.square((-0.237985297) + x[..., 1])
            )
            + 0.175294066
            * numpy.sqrt(
                numpy.square((-0.000770355) + x[..., 0]) + numpy.square((-0.263942898) + x[..., 1])
            )
            + 0.85362463
            * numpy.sqrt(
                numpy.square((-0.608944557) + x[..., 0]) + numpy.square((-0.693859831) + x[..., 1])
            )
            + 0.392007085
            * numpy.sqrt(
                numpy.square((-0.173707857) + x[..., 0]) + numpy.square((-0.092895572) + x[..., 1])
            )
            + 0.516199059
            * numpy.sqrt(
                numpy.square((-0.311185359) + x[..., 0]) + numpy.square((-0.133202159) + x[..., 1])
            )
            + 0.528142598
            * numpy.sqrt(
                numpy.square((-0.728882118) + x[..., 0]) + numpy.square((-0.109171977) + x[..., 1])
            )
            + 0.938252417
            * numpy.sqrt(
                numpy.square((-0.084746778) + x[..., 0]) + numpy.square((-0.833341705) + x[..., 1])
            )
            + 0.380738329
            * numpy.sqrt(
                numpy.square((-0.441986709) + x[..., 0]) + numpy.square((-0.349155773) + x[..., 1])
            )
            + 0.679148954
            * numpy.sqrt(
                numpy.square((-0.659056568) + x[..., 0]) + numpy.square((-0.883199008) + x[..., 1])
            )
            + 0.120042304
            * numpy.sqrt(
                numpy.square((-0.484453602) + x[..., 0]) + numpy.square((-0.851458802) + x[..., 1])
            )
            + 0.327106365
            * numpy.sqrt(
                numpy.square((-0.3181866) + x[..., 0]) + numpy.square((-0.250429003) + x[..., 1])
            )
            + 0.338962533
            * numpy.sqrt(
                numpy.square((-0.914043383) + x[..., 0]) + numpy.square((-0.201071596) + x[..., 1])
            )
            + 0.480061137
            * numpy.sqrt(
                numpy.square((-0.184227426) + x[..., 0]) + numpy.square((-0.65125139) + x[..., 1])
            )
            + 0.986621515
            * numpy.sqrt(
                numpy.square((-0.872113015) + x[..., 0]) + numpy.square((-0.281414622) + x[..., 1])
            )
            + 0.553280512
            * numpy.sqrt(
                numpy.square((-0.456101738) + x[..., 0]) + numpy.square((-0.118585136) + x[..., 1])
            )
        )
        y9 = (
            0.098260179
            * numpy.sqrt(
                numpy.square((-0.458924572) + x[..., 0]) + numpy.square((-0.297739965) + x[..., 1])
            )
            + 0.867410717
            * numpy.sqrt(
                numpy.square((-0.124103467) + x[..., 0]) + numpy.square((-0.483214376) + x[..., 1])
            )
            + 0.942450743
            * numpy.sqrt(
                numpy.square((-0.103803775) + x[..., 0]) + numpy.square((-0.250578007) + x[..., 1])
            )
            + 0.703009491
            * numpy.sqrt(
                numpy.square((-0.23489011) + x[..., 0]) + numpy.square((-0.284357074) + x[..., 1])
            )
            + 0.257285751
            * numpy.sqrt(
                numpy.square((-0.340156838) + x[..., 0]) + numpy.square((-0.814139988) + x[..., 1])
            )
            + 0.998945788
            * numpy.sqrt(
                numpy.square((-0.873065466) + x[..., 0]) + numpy.square((-0.801480037) + x[..., 1])
            )
            + 0.217731555
            * numpy.sqrt(
                numpy.square((-0.844785464) + x[..., 0]) + numpy.square((-0.026432339) + x[..., 1])
            )
            + 0.106451361
            * numpy.sqrt(
                numpy.square((-0.822941662) + x[..., 0]) + numpy.square((-0.312493378) + x[..., 1])
            )
            + 0.908064977
            * numpy.sqrt(
                numpy.square((-0.48051655) + x[..., 0]) + numpy.square((-0.247570852) + x[..., 1])
            )
            + 0.969127305
            * numpy.sqrt(
                numpy.square((-0.913758514) + x[..., 0]) + numpy.square((-0.671522928) + x[..., 1])
            )
            + 0.259160078
            * numpy.sqrt(
                numpy.square((-0.929276607) + x[..., 0]) + numpy.square((-0.836041922) + x[..., 1])
            )
            + 0.132290455
            * numpy.sqrt(
                numpy.square((-0.108108683) + x[..., 0]) + numpy.square((-0.316403195) + x[..., 1])
            )
            + 0.219804543
            * numpy.sqrt(
                numpy.square((-0.350601411) + x[..., 0]) + numpy.square((-0.724203596) + x[..., 1])
            )
            + 0.179069909
            * numpy.sqrt(
                numpy.square((-0.859978394) + x[..., 0]) + numpy.square((-0.841438417) + x[..., 1])
            )
            + 0.114094819
            * numpy.sqrt(
                numpy.square((-0.340504326) + x[..., 0]) + numpy.square((-0.679238269) + x[..., 1])
            )
            + 0.199240513
            * numpy.sqrt(
                numpy.square((-0.213093913) + x[..., 0]) + numpy.square((-0.74009369) + x[..., 1])
            )
            + 0.976486154
            * numpy.sqrt(
                numpy.square((-0.926361531) + x[..., 0]) + numpy.square((-0.529486116) + x[..., 1])
            )
            + 0.433069113
            * numpy.sqrt(
                numpy.square((-0.360725288) + x[..., 0]) + numpy.square((-0.388618354) + x[..., 1])
            )
            + 0.078636719
            * numpy.sqrt(
                numpy.square((-0.875988925) + x[..., 0]) + numpy.square((-0.29609702) + x[..., 1])
            )
            + 0.058475855
            * numpy.sqrt(
                numpy.square((-0.14813862) + x[..., 0]) + numpy.square((-0.932361098) + x[..., 1])
            )
            + 0.274259399
            * numpy.sqrt(
                numpy.square((-0.455984406) + x[..., 0]) + numpy.square((-0.817991343) + x[..., 1])
            )
            + 0.198590288
            * numpy.sqrt(
                numpy.square((-0.268253583) + x[..., 0]) + numpy.square((-0.991563604) + x[..., 1])
            )
            + 0.442961883
            * numpy.sqrt(
                numpy.square((-0.296758523) + x[..., 0]) + numpy.square((-0.450808739) + x[..., 1])
            )
            + 0.838390848
            * numpy.sqrt(
                numpy.square((-0.115496589) + x[..., 0]) + numpy.square((-0.448584782) + x[..., 1])
            )
            + 0.404445585
            * numpy.sqrt(
                numpy.square((-0.399782175) + x[..., 0]) + numpy.square((-0.2723034) + x[..., 1])
            )
            + 0.640897423
            * numpy.sqrt(
                numpy.square((-0.91176171) + x[..., 0]) + numpy.square((-0.881289138) + x[..., 1])
            )
            + 0.894107268
            * numpy.sqrt(
                numpy.square((-0.251551072) + x[..., 0]) + numpy.square((-0.833371988) + x[..., 1])
            )
            + 0.263446938
            * numpy.sqrt(
                numpy.square((-0.102977939) + x[..., 0]) + numpy.square((-0.409643489) + x[..., 1])
            )
            + 0.96419963
            * numpy.sqrt(
                numpy.square((-0.14633273) + x[..., 0]) + numpy.square((-0.053978647) + x[..., 1])
            )
            + 0.558436739
            * numpy.sqrt(
                numpy.square((-0.38630369) + x[..., 0]) + numpy.square((-0.564492285) + x[..., 1])
            )
            + 0.214844307
            * numpy.sqrt(
                numpy.square((-0.046406487) + x[..., 0]) + numpy.square((-0.138119821) + x[..., 1])
            )
            + 0.768645991
            * numpy.sqrt(
                numpy.square((-0.026384119) + x[..., 0]) + numpy.square((-0.423580841) + x[..., 1])
            )
            + 0.841760497
            * numpy.sqrt(
                numpy.square((-0.154000795) + x[..., 0]) + numpy.square((-0.547178851) + x[..., 1])
            )
            + 0.997534317
            * numpy.sqrt(
                numpy.square((-0.072707742) + x[..., 0]) + numpy.square((-0.378451776) + x[..., 1])
            )
            + 0.259276232
            * numpy.sqrt(
                numpy.square((-0.828640826) + x[..., 0]) + numpy.square((-0.567674713) + x[..., 1])
            )
            + 0.694679801
            * numpy.sqrt(
                numpy.square((-0.399774789) + x[..., 0]) + numpy.square((-0.72898962) + x[..., 1])
            )
            + 0.569179453
            * numpy.sqrt(
                numpy.square((-0.417172117) + x[..., 0]) + numpy.square((-0.651407227) + x[..., 1])
            )
            + 0.937940225
            * numpy.sqrt(
                numpy.square((-0.97214974) + x[..., 0]) + numpy.square((-0.246095887) + x[..., 1])
            )
            + 0.260563702
            * numpy.sqrt(
                numpy.square((-0.243431429) + x[..., 0]) + numpy.square((-0.581665105) + x[..., 1])
            )
            + 0.580138483
            * numpy.sqrt(
                numpy.square((-0.361965688) + x[..., 0]) + numpy.square((-0.714589657) + x[..., 1])
            )
            + 0.06038652
            * numpy.sqrt(
                numpy.square((-0.630343873) + x[..., 0]) + numpy.square((-0.188010593) + x[..., 1])
            )
            + 0.953935306
            * numpy.sqrt(
                numpy.square((-0.242852922) + x[..., 0]) + numpy.square((-0.860226263) + x[..., 1])
            )
            + 0.189161234
            * numpy.sqrt(
                numpy.square((-0.101061085) + x[..., 0]) + numpy.square((-0.292306268) + x[..., 1])
            )
            + 0.168512559
            * numpy.sqrt(
                numpy.square((-0.405934329) + x[..., 0]) + numpy.square((-0.218619166) + x[..., 1])
            )
            + 0.831237135
            * numpy.sqrt(
                numpy.square((-0.479057225) + x[..., 0]) + numpy.square((-0.51142729) + x[..., 1])
            )
            + 0.11067703
            * numpy.sqrt(
                numpy.square((-0.144946363) + x[..., 0]) + numpy.square((-0.699105435) + x[..., 1])
            )
            + 0.038899269
            * numpy.sqrt(
                numpy.square((-0.509687177) + x[..., 0]) + numpy.square((-0.56026682) + x[..., 1])
            )
            + 0.060781843
            * numpy.sqrt(
                numpy.square((-0.885281305) + x[..., 0]) + numpy.square((-0.336157322) + x[..., 1])
            )
            + 0.421373907
            * numpy.sqrt(
                numpy.square((-0.055478358) + x[..., 0]) + numpy.square((-0.518764995) + x[..., 1])
            )
            + 0.895030215
            * numpy.sqrt(
                numpy.square((-0.507403592) + x[..., 0]) + numpy.square((-0.249221613) + x[..., 1])
            )
            + 0.660637595
            * numpy.sqrt(
                numpy.square((-0.764113651) + x[..., 0]) + numpy.square((-0.410963477) + x[..., 1])
            )
            + 0.580902193
            * numpy.sqrt(
                numpy.square((-0.979001752) + x[..., 0]) + numpy.square((-0.516282412) + x[..., 1])
            )
            + 0.809613447
            * numpy.sqrt(
                numpy.square((-0.72041588) + x[..., 0]) + numpy.square((-0.768859763) + x[..., 1])
            )
            + 0.481912049
            * numpy.sqrt(
                numpy.square((-0.871947756) + x[..., 0]) + numpy.square((-0.663947588) + x[..., 1])
            )
            + 0.095627993
            * numpy.sqrt(
                numpy.square((-0.299524923) + x[..., 0]) + numpy.square((-0.019156127) + x[..., 1])
            )
            + 0.372963548
            * numpy.sqrt(
                numpy.square((-0.753895467) + x[..., 0]) + numpy.square((-0.908635573) + x[..., 1])
            )
            + 0.346555598
            * numpy.sqrt(
                numpy.square((-0.843756424) + x[..., 0]) + numpy.square((-0.45338404) + x[..., 1])
            )
            + 0.37544984
            * numpy.sqrt(
                numpy.square((-0.468227734) + x[..., 0]) + numpy.square((-0.667635022) + x[..., 1])
            )
            + 0.778525086
            * numpy.sqrt(
                numpy.square((-0.654929517) + x[..., 0]) + numpy.square((-0.075785181) + x[..., 1])
            )
            + 0.773147258
            * numpy.sqrt(
                numpy.square((-0.378051534) + x[..., 0]) + numpy.square((-0.411412731) + x[..., 1])
            )
            + 0.427130564
            * numpy.sqrt(
                numpy.square((-0.358874791) + x[..., 0]) + numpy.square((-0.518402374) + x[..., 1])
            )
            + 0.55616776
            * numpy.sqrt(
                numpy.square((-0.254480715) + x[..., 0]) + numpy.square((-0.006897654) + x[..., 1])
            )
            + 0.436065549
            * numpy.sqrt(
                numpy.square((-0.255482115) + x[..., 0]) + numpy.square((-0.186652502) + x[..., 1])
            )
            + 0.719748882
            * numpy.sqrt(
                numpy.square((-0.450619222) + x[..., 0]) + numpy.square((-0.183606377) + x[..., 1])
            )
            + 0.596143148
            * numpy.sqrt(
                numpy.square((-0.944854547) + x[..., 0]) + numpy.square((-0.543649159) + x[..., 1])
            )
            + 0.22135876
            * numpy.sqrt(
                numpy.square((-0.335525891) + x[..., 0]) + numpy.square((-0.552724283) + x[..., 1])
            )
            + 0.664463396
            * numpy.sqrt(
                numpy.square((-0.048490732) + x[..., 0]) + numpy.square((-0.401553563) + x[..., 1])
            )
            + 0.223494526
            * numpy.sqrt(
                numpy.square((-0.80649476) + x[..., 0]) + numpy.square((-0.874276406) + x[..., 1])
            )
            + 0.684961924
            * numpy.sqrt(
                numpy.square((-0.732614118) + x[..., 0]) + numpy.square((-0.621774869) + x[..., 1])
            )
            + 0.417465568
            * numpy.sqrt(
                numpy.square((-0.920014559) + x[..., 0]) + numpy.square((-0.410066542) + x[..., 1])
            )
            + 0.899353148
            * numpy.sqrt(
                numpy.square((-0.331644775) + x[..., 0]) + numpy.square((-0.273074885) + x[..., 1])
            )
            + 0.319956486
            * numpy.sqrt(
                numpy.square((-0.209776172) + x[..., 0]) + numpy.square((-0.012998451) + x[..., 1])
            )
            + 0.257775047
            * numpy.sqrt(
                numpy.square((-0.047148973) + x[..., 0]) + numpy.square((-0.364434013) + x[..., 1])
            )
            + 0.225012089
            * numpy.sqrt(
                numpy.square((-0.883036523) + x[..., 0]) + numpy.square((-0.602237189) + x[..., 1])
            )
            + 0.66646877
            * numpy.sqrt(
                numpy.square((-0.092805702) + x[..., 0]) + numpy.square((-0.839465526) + x[..., 1])
            )
            + 0.75806034
            * numpy.sqrt(
                numpy.square((-0.836905575) + x[..., 0]) + numpy.square((-0.684789184) + x[..., 1])
            )
            + 0.67044769
            * numpy.sqrt(
                numpy.square((-0.463811546) + x[..., 0]) + numpy.square((-0.882391627) + x[..., 1])
            )
            + 0.504709466
            * numpy.sqrt(
                numpy.square((-0.000165259) + x[..., 0]) + numpy.square((-0.617467034) + x[..., 1])
            )
            + 0.457652519
            * numpy.sqrt(
                numpy.square((-0.156576815) + x[..., 0]) + numpy.square((-0.099429009) + x[..., 1])
            )
            + 0.725296631
            * numpy.sqrt(
                numpy.square((-0.705029105) + x[..., 0]) + numpy.square((-0.389104392) + x[..., 1])
            )
            + 0.520855119
            * numpy.sqrt(
                numpy.square((-0.280347043) + x[..., 0]) + numpy.square((-0.029830336) + x[..., 1])
            )
            + 0.567000903
            * numpy.sqrt(
                numpy.square((-0.635576041) + x[..., 0]) + numpy.square((-0.582942985) + x[..., 1])
            )
            + 0.924484405
            * numpy.sqrt(
                numpy.square((-0.579960593) + x[..., 0]) + numpy.square((-0.208571893) + x[..., 1])
            )
            + 0.484138493
            * numpy.sqrt(
                numpy.square((-0.962105304) + x[..., 0]) + numpy.square((-0.402727257) + x[..., 1])
            )
            + 0.571815224
            * numpy.sqrt(
                numpy.square((-0.514190675) + x[..., 0]) + numpy.square((-0.283077663) + x[..., 1])
            )
            + 0.661563795
            * numpy.sqrt(
                numpy.square((-0.590304749) + x[..., 0]) + numpy.square((-0.019534685) + x[..., 1])
            )
            + 0.670385812
            * numpy.sqrt(
                numpy.square((-0.567459547) + x[..., 0]) + numpy.square((-0.125840876) + x[..., 1])
            )
            + 0.121029803
            * numpy.sqrt(
                numpy.square((-0.029897328) + x[..., 0]) + numpy.square((-0.063964475) + x[..., 1])
            )
            + 0.528447785
            * numpy.sqrt(
                numpy.square((-0.968903967) + x[..., 0]) + numpy.square((-0.127873769) + x[..., 1])
            )
            + 0.824333358
            * numpy.sqrt(
                numpy.square((-0.593750716) + x[..., 0]) + numpy.square((-0.283317639) + x[..., 1])
            )
            + 0.596057589
            * numpy.sqrt(
                numpy.square((-0.059617951) + x[..., 0]) + numpy.square((-0.914849632) + x[..., 1])
            )
            + 0.231952064
            * numpy.sqrt(
                numpy.square((-0.544106651) + x[..., 0]) + numpy.square((-0.84992719) + x[..., 1])
            )
            + 0.947708267
            * numpy.sqrt(
                numpy.square((-0.127364276) + x[..., 0]) + numpy.square((-0.219663548) + x[..., 1])
            )
            + 0.121920417
            * numpy.sqrt(
                numpy.square((-0.420489767) + x[..., 0]) + numpy.square((-0.269171727) + x[..., 1])
            )
            + 0.702150098
            * numpy.sqrt(
                numpy.square((-0.881449138) + x[..., 0]) + numpy.square((-0.467018805) + x[..., 1])
            )
            + 0.461777689
            * numpy.sqrt(
                numpy.square((-0.716585359) + x[..., 0]) + numpy.square((-0.516487665) + x[..., 1])
            )
            + 0.991852778
            * numpy.sqrt(
                numpy.square((-0.313576315) + x[..., 0]) + numpy.square((-0.024184341) + x[..., 1])
            )
            + 0.219377639
            * numpy.sqrt(
                numpy.square((-0.992952402) + x[..., 0]) + numpy.square((-0.394189606) + x[..., 1])
            )
            + 0.514197486
            * numpy.sqrt(
                numpy.square((-0.055332918) + x[..., 0]) + numpy.square((-0.675867576) + x[..., 1])
            )
            + 0.084215436
            * numpy.sqrt(
                numpy.square((-0.619078831) + x[..., 0]) + numpy.square((-0.102418315) + x[..., 1])
            )
            + 0.953775164
            * numpy.sqrt(
                numpy.square((-0.804442037) + x[..., 0]) + numpy.square((-0.371558118) + x[..., 1])
            )
            + 0.719684604
            * numpy.sqrt(
                numpy.square((-0.296593264) + x[..., 0]) + numpy.square((-0.570604363) + x[..., 1])
            )
            + 0.801211877
            * numpy.sqrt(
                numpy.square((-0.958919774) + x[..., 0]) + numpy.square((-0.295997377) + x[..., 1])
            )
            + 0.117534166
            * numpy.sqrt(
                numpy.square((-0.69475307) + x[..., 0]) + numpy.square((-0.62579104) + x[..., 1])
            )
            + 0.85917936
            * numpy.sqrt(
                numpy.square((-0.631414667) + x[..., 0]) + numpy.square((-0.939290692) + x[..., 1])
            )
            + 0.162681384
            * numpy.sqrt(
                numpy.square((-0.615975031) + x[..., 0]) + numpy.square((-0.790117117) + x[..., 1])
            )
            + 0.654078988
            * numpy.sqrt(
                numpy.square((-0.523017346) + x[..., 0]) + numpy.square((-0.342920518) + x[..., 1])
            )
            + 0.249842787
            * numpy.sqrt(
                numpy.square((-0.141031812) + x[..., 0]) + numpy.square((-0.585573571) + x[..., 1])
            )
            + 0.016101852
            * numpy.sqrt(
                numpy.square((-0.836730986) + x[..., 0]) + numpy.square((-0.606747374) + x[..., 1])
            )
            + 0.698757317
            * numpy.sqrt(
                numpy.square((-0.89209171) + x[..., 0]) + numpy.square((-0.25588444) + x[..., 1])
            )
            + 0.069040824
            * numpy.sqrt(
                numpy.square((-0.696470459) + x[..., 0]) + numpy.square((-0.038296176) + x[..., 1])
            )
            + 0.720730694
            * numpy.sqrt(
                numpy.square((-0.311787525) + x[..., 0]) + numpy.square((-0.672775317) + x[..., 1])
            )
        )
        y10 = (
            0.469086532
            * numpy.sqrt(
                numpy.square((-0.355821022) + x[..., 0]) + numpy.square((-0.495655514) + x[..., 1])
            )
            + 0.434942874
            * numpy.sqrt(
                numpy.square((-0.761080695) + x[..., 0]) + numpy.square((-0.3834957) + x[..., 1])
            )
            + 0.098622212
            * numpy.sqrt(
                numpy.square((-0.13636211) + x[..., 0]) + numpy.square((-0.777545237) + x[..., 1])
            )
            + 0.337227389
            * numpy.sqrt(
                numpy.square((-0.716631872) + x[..., 0]) + numpy.square((-0.899272299) + x[..., 1])
            )
            + 0.990231958
            * numpy.sqrt(
                numpy.square((-0.963700889) + x[..., 0]) + numpy.square((-0.006519096) + x[..., 1])
            )
            + 0.138643285
            * numpy.sqrt(
                numpy.square((-0.441918035) + x[..., 0]) + numpy.square((-0.969447199) + x[..., 1])
            )
            + 0.877988385
            * numpy.sqrt(
                numpy.square((-0.264478318) + x[..., 0]) + numpy.square((-0.573025942) + x[..., 1])
            )
            + 0.598608831
            * numpy.sqrt(
                numpy.square((-0.444178314) + x[..., 0]) + numpy.square((-0.083410412) + x[..., 1])
            )
            + 0.962082528
            * numpy.sqrt(
                numpy.square((-0.396704692) + x[..., 0]) + numpy.square((-0.345191825) + x[..., 1])
            )
            + 0.969783595
            * numpy.sqrt(
                numpy.square((-0.366924315) + x[..., 0]) + numpy.square((-0.898116662) + x[..., 1])
            )
            + 0.654315073
            * numpy.sqrt(numpy.square((-0.6212629) + x[..., 0]) + numpy.square((-0.54275159) + x[..., 1]))
            + 0.747021699
            * numpy.sqrt(
                numpy.square((-0.859730557) + x[..., 0]) + numpy.square((-0.409008975) + x[..., 1])
            )
            + 0.156901866
            * numpy.sqrt(
                numpy.square((-0.755006907) + x[..., 0]) + numpy.square((-0.327547317) + x[..., 1])
            )
            + 0.319214665
            * numpy.sqrt(
                numpy.square((-0.900814859) + x[..., 0]) + numpy.square((-0.989662457) + x[..., 1])
            )
            + 0.259569966
            * numpy.sqrt(
                numpy.square((-0.43804536) + x[..., 0]) + numpy.square((-0.040159945) + x[..., 1])
            )
            + 0.303839741
            * numpy.sqrt(
                numpy.square((-0.069098549) + x[..., 0]) + numpy.square((-0.892973955) + x[..., 1])
            )
            + 0.035850855
            * numpy.sqrt(
                numpy.square((-0.938574853) + x[..., 0]) + numpy.square((-0.309930921) + x[..., 1])
            )
            + 0.497956211
            * numpy.sqrt(
                numpy.square((-0.843967544) + x[..., 0]) + numpy.square((-0.62084636) + x[..., 1])
            )
            + 0.926823205
            * numpy.sqrt(
                numpy.square((-0.950440862) + x[..., 0]) + numpy.square((-0.173361894) + x[..., 1])
            )
            + 0.55977066
            * numpy.sqrt(
                numpy.square((-0.579384068) + x[..., 0]) + numpy.square((-0.183286192) + x[..., 1])
            )
            + 0.465810197
            * numpy.sqrt(
                numpy.square((-0.03521677) + x[..., 0]) + numpy.square((-0.057200214) + x[..., 1])
            )
            + 0.396870676
            * numpy.sqrt(
                numpy.square((-0.407433213) + x[..., 0]) + numpy.square((-0.763271687) + x[..., 1])
            )
            + 0.303922724
            * numpy.sqrt(
                numpy.square((-0.057425008) + x[..., 0]) + numpy.square((-0.870339397) + x[..., 1])
            )
            + 0.625824904
            * numpy.sqrt(
                numpy.square((-0.531968899) + x[..., 0]) + numpy.square((-0.780608723) + x[..., 1])
            )
            + 0.906363308
            * numpy.sqrt(
                numpy.square((-0.112406712) + x[..., 0]) + numpy.square((-0.171979508) + x[..., 1])
            )
            + 0.343582212
            * numpy.sqrt(
                numpy.square((-0.893848822) + x[..., 0]) + numpy.square((-0.692289481) + x[..., 1])
            )
            + 0.674525046
            * numpy.sqrt(
                numpy.square((-0.241689424) + x[..., 0]) + numpy.square((-0.633867494) + x[..., 1])
            )
            + 0.089903175
            * numpy.sqrt(
                numpy.square((-0.642242919) + x[..., 0]) + numpy.square((-0.619130075) + x[..., 1])
            )
            + 0.565935802
            * numpy.sqrt(
                numpy.square((-0.290779826) + x[..., 0]) + numpy.square((-0.000378558) + x[..., 1])
            )
            + 0.297422264
            * numpy.sqrt(
                numpy.square((-0.81356408) + x[..., 0]) + numpy.square((-0.217205303) + x[..., 1])
            )
            + 0.166944653
            * numpy.sqrt(
                numpy.square((-0.186140904) + x[..., 0]) + numpy.square((-0.610580435) + x[..., 1])
            )
            + 0.803043177
            * numpy.sqrt(
                numpy.square((-0.500676233) + x[..., 0]) + numpy.square((-0.460238747) + x[..., 1])
            )
            + 0.078893708
            * numpy.sqrt(
                numpy.square((-0.938821199) + x[..., 0]) + numpy.square((-0.207532458) + x[..., 1])
            )
            + 0.922117636
            * numpy.sqrt(
                numpy.square((-0.681566417) + x[..., 0]) + numpy.square((-0.503436901) + x[..., 1])
            )
            + 0.968525998
            * numpy.sqrt(
                numpy.square((-0.18523186) + x[..., 0]) + numpy.square((-0.331524816) + x[..., 1])
            )
            + 0.824654794
            * numpy.sqrt(
                numpy.square((-0.872883561) + x[..., 0]) + numpy.square((-0.212765894) + x[..., 1])
            )
            + 0.211658572
            * numpy.sqrt(
                numpy.square((-0.165007652) + x[..., 0]) + numpy.square((-0.082519225) + x[..., 1])
            )
            + 0.945191516
            * numpy.sqrt(
                numpy.square((-0.430829916) + x[..., 0]) + numpy.square((-0.283412428) + x[..., 1])
            )
            + 0.391616643
            * numpy.sqrt(
                numpy.square((-0.771730812) + x[..., 0]) + numpy.square((-0.972838506) + x[..., 1])
            )
            + 0.854103608
            * numpy.sqrt(
                numpy.square((-0.978050424) + x[..., 0]) + numpy.square((-0.471460058) + x[..., 1])
            )
            + 0.845651633
            * numpy.sqrt(
                numpy.square((-0.94446167) + x[..., 0]) + numpy.square((-0.221964608) + x[..., 1])
            )
            + 0.091834477
            * numpy.sqrt(
                numpy.square((-0.248838158) + x[..., 0]) + numpy.square((-0.565368503) + x[..., 1])
            )
            + 0.284902004
            * numpy.sqrt(
                numpy.square((-0.886464713) + x[..., 0]) + numpy.square((-0.899656138) + x[..., 1])
            )
            + 0.918793715
            * numpy.sqrt(
                numpy.square((-0.884427478) + x[..., 0]) + numpy.square((-0.157082827) + x[..., 1])
            )
            + 0.102620037
            * numpy.sqrt(
                numpy.square((-0.964948489) + x[..., 0]) + numpy.square((-0.291328392) + x[..., 1])
            )
            + 0.0034196
            * numpy.sqrt(
                numpy.square((-0.749143602) + x[..., 0]) + numpy.square((-0.145265526) + x[..., 1])
            )
            + 0.805547203
            * numpy.sqrt(
                numpy.square((-0.647604097) + x[..., 0]) + numpy.square((-0.859085318) + x[..., 1])
            )
            + 0.042706144
            * numpy.sqrt(
                numpy.square((-0.748378144) + x[..., 0]) + numpy.square((-0.023549377) + x[..., 1])
            )
            + 0.178922659
            * numpy.sqrt(
                numpy.square((-0.523169329) + x[..., 0]) + numpy.square((-0.845138333) + x[..., 1])
            )
            + 0.811471219
            * numpy.sqrt(
                numpy.square((-0.686111771) + x[..., 0]) + numpy.square((-0.271976939) + x[..., 1])
            )
            + 0.643847291
            * numpy.sqrt(
                numpy.square((-0.691068315) + x[..., 0]) + numpy.square((-0.773845497) + x[..., 1])
            )
            + 0.269976259
            * numpy.sqrt(
                numpy.square((-0.865647846) + x[..., 0]) + numpy.square((-0.120151523) + x[..., 1])
            )
            + 0.103098518
            * numpy.sqrt(
                numpy.square((-0.795756739) + x[..., 0]) + numpy.square((-0.78861013) + x[..., 1])
            )
            + 0.882652266
            * numpy.sqrt(
                numpy.square((-0.217550999) + x[..., 0]) + numpy.square((-0.793747819) + x[..., 1])
            )
            + 0.95408515
            * numpy.sqrt(
                numpy.square((-0.589621917) + x[..., 0]) + numpy.square((-0.831457286) + x[..., 1])
            )
            + 0.725404608
            * numpy.sqrt(
                numpy.square((-0.922616095) + x[..., 0]) + numpy.square((-0.113793123) + x[..., 1])
            )
            + 0.74543965
            * numpy.sqrt(
                numpy.square((-0.602902782) + x[..., 0]) + numpy.square((-0.034126737) + x[..., 1])
            )
            + 0.673939953
            * numpy.sqrt(
                numpy.square((-0.035704498) + x[..., 0]) + numpy.square((-0.729251314) + x[..., 1])
            )
            + 0.263426066
            * numpy.sqrt(
                numpy.square((-0.911119939) + x[..., 0]) + numpy.square((-0.886889598) + x[..., 1])
            )
            + 0.123053473
            * numpy.sqrt(
                numpy.square((-0.565012773) + x[..., 0]) + numpy.square((-0.262024978) + x[..., 1])
            )
            + 0.910261845
            * numpy.sqrt(
                numpy.square((-0.324824414) + x[..., 0]) + numpy.square((-0.790538072) + x[..., 1])
            )
            + 0.756958582
            * numpy.sqrt(
                numpy.square((-0.390124939) + x[..., 0]) + numpy.square((-0.635220824) + x[..., 1])
            )
            + 0.070161046
            * numpy.sqrt(
                numpy.square((-0.299323103) + x[..., 0]) + numpy.square((-0.118827179) + x[..., 1])
            )
            + 0.13774917
            * numpy.sqrt(
                numpy.square((-0.218964188) + x[..., 0]) + numpy.square((-0.521155067) + x[..., 1])
            )
            + 0.373726006
            * numpy.sqrt(
                numpy.square((-0.821720008) + x[..., 0]) + numpy.square((-0.172224998) + x[..., 1])
            )
            + 0.422363009
            * numpy.sqrt(
                numpy.square((-0.152662912) + x[..., 0]) + numpy.square((-0.110499073) + x[..., 1])
            )
            + 0.799313432
            * numpy.sqrt(
                numpy.square((-0.950547885) + x[..., 0]) + numpy.square((-0.202762177) + x[..., 1])
            )
            + 0.034981449
            * numpy.sqrt(
                numpy.square((-0.03191729) + x[..., 0]) + numpy.square((-0.214475291) + x[..., 1])
            )
            + 0.617064567
            * numpy.sqrt(
                numpy.square((-0.734463261) + x[..., 0]) + numpy.square((-0.279275971) + x[..., 1])
            )
            + 0.797792277
            * numpy.sqrt(
                numpy.square((-0.110492862) + x[..., 0]) + numpy.square((-0.150731851) + x[..., 1])
            )
            + 0.311986963
            * numpy.sqrt(
                numpy.square((-0.487752817) + x[..., 0]) + numpy.square((-0.766705722) + x[..., 1])
            )
            + 0.816473216
            * numpy.sqrt(
                numpy.square((-0.361037498) + x[..., 0]) + numpy.square((-0.915741315) + x[..., 1])
            )
            + 0.983758725
            * numpy.sqrt(
                numpy.square((-0.216392286) + x[..., 0]) + numpy.square((-0.944978784) + x[..., 1])
            )
            + 0.842607769
            * numpy.sqrt(
                numpy.square((-0.923706706) + x[..., 0]) + numpy.square((-0.070943579) + x[..., 1])
            )
            + 0.605995496
            * numpy.sqrt(
                numpy.square((-0.449963497) + x[..., 0]) + numpy.square((-0.811728074) + x[..., 1])
            )
            + 0.746125593
            * numpy.sqrt(
                numpy.square((-0.97108314) + x[..., 0]) + numpy.square((-0.486595851) + x[..., 1])
            )
            + 0.595658799
            * numpy.sqrt(
                numpy.square((-0.096334971) + x[..., 0]) + numpy.square((-0.46727431) + x[..., 1])
            )
            + 0.186388558
            * numpy.sqrt(
                numpy.square((-0.47891946) + x[..., 0]) + numpy.square((-0.244548357) + x[..., 1])
            )
            + 0.766645544
            * numpy.sqrt(
                numpy.square((-0.722165608) + x[..., 0]) + numpy.square((-0.841318031) + x[..., 1])
            )
            + 0.743681924
            * numpy.sqrt(
                numpy.square((-0.433204927) + x[..., 0]) + numpy.square((-0.381862391) + x[..., 1])
            )
            + 0.689503224
            * numpy.sqrt(
                numpy.square((-0.158177913) + x[..., 0]) + numpy.square((-0.349368963) + x[..., 1])
            )
            + 0.424682259
            * numpy.sqrt(
                numpy.square((-0.100657612) + x[..., 0]) + numpy.square((-0.048293923) + x[..., 1])
            )
            + 0.073801323
            * numpy.sqrt(
                numpy.square((-0.805511933) + x[..., 0]) + numpy.square((-0.012382701) + x[..., 1])
            )
            + 0.867178485
            * numpy.sqrt(
                numpy.square((-0.398688116) + x[..., 0]) + numpy.square((-0.744494118) + x[..., 1])
            )
            + 0.22838144
            * numpy.sqrt(
                numpy.square((-0.117093621) + x[..., 0]) + numpy.square((-0.178909127) + x[..., 1])
            )
            + 0.936798903
            * numpy.sqrt(
                numpy.square((-0.874353379) + x[..., 0]) + numpy.square((-0.937805118) + x[..., 1])
            )
            + 0.642397834
            * numpy.sqrt(
                numpy.square((-0.144855211) + x[..., 0]) + numpy.square((-0.964583233) + x[..., 1])
            )
            + 0.211486063
            * numpy.sqrt(
                numpy.square((-0.177740504) + x[..., 0]) + numpy.square((-0.824036758) + x[..., 1])
            )
            + 0.029012651
            * numpy.sqrt(
                numpy.square((-0.545204307) + x[..., 0]) + numpy.square((-0.725891664) + x[..., 1])
            )
            + 0.508924996
            * numpy.sqrt(
                numpy.square((-0.468599988) + x[..., 0]) + numpy.square((-0.432252517) + x[..., 1])
            )
            + 0.465681288
            * numpy.sqrt(
                numpy.square((-0.909182672) + x[..., 0]) + numpy.square((-0.206246798) + x[..., 1])
            )
            + 0.647125901
            * numpy.sqrt(
                numpy.square((-0.723089617) + x[..., 0]) + numpy.square((-0.113072026) + x[..., 1])
            )
            + 0.355177587
            * numpy.sqrt(
                numpy.square((-0.166351598) + x[..., 0]) + numpy.square((-0.710386486) + x[..., 1])
            )
            + 0.124498774
            * numpy.sqrt(
                numpy.square((-0.327552498) + x[..., 0]) + numpy.square((-0.889428094) + x[..., 1])
            )
            + 0.795248146
            * numpy.sqrt(
                numpy.square((-0.581345732) + x[..., 0]) + numpy.square((-0.126268233) + x[..., 1])
            )
            + 0.394043263
            * numpy.sqrt(
                numpy.square((-0.577537355) + x[..., 0]) + numpy.square((-0.04637996) + x[..., 1])
            )
            + 0.637997466
            * numpy.sqrt(
                numpy.square((-0.627575258) + x[..., 0]) + numpy.square((-0.000232349) + x[..., 1])
            )
            + 0.310243366
            * numpy.sqrt(
                numpy.square((-0.026734156) + x[..., 0]) + numpy.square((-0.641115781) + x[..., 1])
            )
            + 0.095470838
            * numpy.sqrt(
                numpy.square((-0.129420571) + x[..., 0]) + numpy.square((-0.26514149) + x[..., 1])
            )
            + 0.430746986
            * numpy.sqrt(
                numpy.square((-0.06413363) + x[..., 0]) + numpy.square((-0.087218083) + x[..., 1])
            )
            + 0.597243966
            * numpy.sqrt(
                numpy.square((-0.311098242) + x[..., 0]) + numpy.square((-0.76513459) + x[..., 1])
            )
            + 0.347663848
            * numpy.sqrt(
                numpy.square((-0.578505548) + x[..., 0]) + numpy.square((-0.697492762) + x[..., 1])
            )
            + 0.071884561
            * numpy.sqrt(
                numpy.square((-0.809803291) + x[..., 0]) + numpy.square((-0.32977267) + x[..., 1])
            )
            + 0.286998411
            * numpy.sqrt(
                numpy.square((-0.679201785) + x[..., 0]) + numpy.square((-0.465528052) + x[..., 1])
            )
            + 0.253765454
            * numpy.sqrt(
                numpy.square((-0.735670909) + x[..., 0]) + numpy.square((-0.100689498) + x[..., 1])
            )
            + 0.95664551
            * numpy.sqrt(
                numpy.square((-0.338608099) + x[..., 0]) + numpy.square((-0.11920025) + x[..., 1])
            )
            + 0.626470703
            * numpy.sqrt(
                numpy.square((-0.224235075) + x[..., 0]) + numpy.square((-0.924884064) + x[..., 1])
            )
            + 0.510859319
            * numpy.sqrt(
                numpy.square((-0.900027148) + x[..., 0]) + numpy.square((-0.466601346) + x[..., 1])
            )
            + 0.501810911
            * numpy.sqrt(
                numpy.square((-0.82938427) + x[..., 0]) + numpy.square((-0.40828478) + x[..., 1])
            )
            + 0.23014575
            * numpy.sqrt(
                numpy.square((-0.316222102) + x[..., 0]) + numpy.square((-0.013688632) + x[..., 1])
            )
            + 0.160645343
            * numpy.sqrt(
                numpy.square((-0.952220667) + x[..., 0]) + numpy.square((-0.577069065) + x[..., 1])
            )
            + 0.867407645
            * numpy.sqrt(
                numpy.square((-0.256689072) + x[..., 0]) + numpy.square((-0.956734713) + x[..., 1])
            )
        )
        y11 = (
            0.893104985
            * numpy.sqrt(
                numpy.square((-0.626115201) + x[..., 0]) + numpy.square((-0.454802888) + x[..., 1])
            )
            + 0.459601002
            * numpy.sqrt(
                numpy.square((-0.971255852) + x[..., 0]) + numpy.square((-0.99465842) + x[..., 1])
            )
            + 0.358973506
            * numpy.sqrt(
                numpy.square((-0.962081396) + x[..., 0]) + numpy.square((-0.30614099) + x[..., 1])
            )
            + 0.564532671
            * numpy.sqrt(
                numpy.square((-0.425281792) + x[..., 0]) + numpy.square((-0.405376764) + x[..., 1])
            )
            + 0.590013929
            * numpy.sqrt(
                numpy.square((-0.105393477) + x[..., 0]) + numpy.square((-0.528872992) + x[..., 1])
            )
            + 0.267117869
            * numpy.sqrt(
                numpy.square((-0.077083987) + x[..., 0]) + numpy.square((-0.268037069) + x[..., 1])
            )
            + 0.058855462
            * numpy.sqrt(
                numpy.square((-0.644126466) + x[..., 0]) + numpy.square((-0.731245046) + x[..., 1])
            )
            + 0.45160837
            * numpy.sqrt(
                numpy.square((-0.312220199) + x[..., 0]) + numpy.square((-0.175584648) + x[..., 1])
            )
            + 0.453837011
            * numpy.sqrt(
                numpy.square((-0.595182863) + x[..., 0]) + numpy.square((-0.565136154) + x[..., 1])
            )
            + 0.974658638
            * numpy.sqrt(
                numpy.square((-0.606377986) + x[..., 0]) + numpy.square((-0.258540357) + x[..., 1])
            )
            + 0.823322877
            * numpy.sqrt(
                numpy.square((-0.633652087) + x[..., 0]) + numpy.square((-0.891941337) + x[..., 1])
            )
            + 0.671873051
            * numpy.sqrt(
                numpy.square((-0.958236602) + x[..., 0]) + numpy.square((-0.204110309) + x[..., 1])
            )
            + 0.406175634
            * numpy.sqrt(
                numpy.square((-0.082263467) + x[..., 0]) + numpy.square((-0.380130936) + x[..., 1])
            )
            + 0.457883355
            * numpy.sqrt(
                numpy.square((-0.125371273) + x[..., 0]) + numpy.square((-0.161592556) + x[..., 1])
            )
            + 0.970821583
            * numpy.sqrt(
                numpy.square((-0.605220151) + x[..., 0]) + numpy.square((-0.693776707) + x[..., 1])
            )
            + 0.891924248
            * numpy.sqrt(
                numpy.square((-0.74147994) + x[..., 0]) + numpy.square((-0.379610281) + x[..., 1])
            )
            + 0.557705765
            * numpy.sqrt(
                numpy.square((-0.847520435) + x[..., 0]) + numpy.square((-0.171287441) + x[..., 1])
            )
            + 0.687201781
            * numpy.sqrt(
                numpy.square((-0.352460216) + x[..., 0]) + numpy.square((-0.654645247) + x[..., 1])
            )
            + 0.643505556
            * numpy.sqrt(
                numpy.square((-0.641412771) + x[..., 0]) + numpy.square((-0.195267459) + x[..., 1])
            )
            + 0.592239199
            * numpy.sqrt(
                numpy.square((-0.895729591) + x[..., 0]) + numpy.square((-0.322008784) + x[..., 1])
            )
            + 0.380841355
            * numpy.sqrt(
                numpy.square((-0.388166729) + x[..., 0]) + numpy.square((-0.385463039) + x[..., 1])
            )
            + 0.7239394
            * numpy.sqrt(
                numpy.square((-0.273399815) + x[..., 0]) + numpy.square((-0.817768654) + x[..., 1])
            )
            + 0.564759265
            * numpy.sqrt(
                numpy.square((-0.970395428) + x[..., 0]) + numpy.square((-0.536724058) + x[..., 1])
            )
            + 0.878032005
            * numpy.sqrt(
                numpy.square((-0.34621371) + x[..., 0]) + numpy.square((-0.077392024) + x[..., 1])
            )
            + 0.780834968
            * numpy.sqrt(
                numpy.square((-0.409589297) + x[..., 0]) + numpy.square((-0.274448426) + x[..., 1])
            )
            + 0.803825064
            * numpy.sqrt(
                numpy.square((-0.939864087) + x[..., 0]) + numpy.square((-0.892231178) + x[..., 1])
            )
            + 0.764079967
            * numpy.sqrt(
                numpy.square((-0.602931377) + x[..., 0]) + numpy.square((-0.955883988) + x[..., 1])
            )
            + 0.345729174
            * numpy.sqrt(
                numpy.square((-0.899542622) + x[..., 0]) + numpy.square((-0.385161308) + x[..., 1])
            )
            + 0.358707992
            * numpy.sqrt(
                numpy.square((-0.284731462) + x[..., 0]) + numpy.square((-0.106347832) + x[..., 1])
            )
            + 0.759513345
            * numpy.sqrt(
                numpy.square((-0.222239116) + x[..., 0]) + numpy.square((-0.85079011) + x[..., 1])
            )
            + 0.968344969
            * numpy.sqrt(
                numpy.square((-0.574837722) + x[..., 0]) + numpy.square((-0.789910021) + x[..., 1])
            )
            + 0.960097124
            * numpy.sqrt(
                numpy.square((-0.509499809) + x[..., 0]) + numpy.square((-0.99664035) + x[..., 1])
            )
            + 0.226678355
            * numpy.sqrt(
                numpy.square((-0.557480049) + x[..., 0]) + numpy.square((-0.545362919) + x[..., 1])
            )
            + 0.35992843
            * numpy.sqrt(
                numpy.square((-0.344169899) + x[..., 0]) + numpy.square((-0.94422182) + x[..., 1])
            )
            + 0.926511493
            * numpy.sqrt(
                numpy.square((-0.398265203) + x[..., 0]) + numpy.square((-0.322533954) + x[..., 1])
            )
            + 0.701962561
            * numpy.sqrt(
                numpy.square((-0.776226659) + x[..., 0]) + numpy.square((-0.075165662) + x[..., 1])
            )
            + 0.499778739
            * numpy.sqrt(
                numpy.square((-0.028229395) + x[..., 0]) + numpy.square((-0.770600915) + x[..., 1])
            )
            + 0.947878747
            * numpy.sqrt(
                numpy.square((-0.362381544) + x[..., 0]) + numpy.square((-0.605113166) + x[..., 1])
            )
            + 0.971447544
            * numpy.sqrt(
                numpy.square((-0.755817347) + x[..., 0]) + numpy.square((-0.412632774) + x[..., 1])
            )
            + 0.693085451
            * numpy.sqrt(
                numpy.square((-0.474912346) + x[..., 0]) + numpy.square((-0.985800694) + x[..., 1])
            )
            + 0.801266016
            * numpy.sqrt(
                numpy.square((-0.076186271) + x[..., 0]) + numpy.square((-0.020656254) + x[..., 1])
            )
            + 0.919273862
            * numpy.sqrt(
                numpy.square((-0.097503281) + x[..., 0]) + numpy.square((-0.290357343) + x[..., 1])
            )
            + 0.114051522
            * numpy.sqrt(
                numpy.square((-0.329670369) + x[..., 0]) + numpy.square((-0.429400444) + x[..., 1])
            )
            + 0.315567106
            * numpy.sqrt(numpy.square((-0.20060538) + x[..., 0]) + numpy.square((-0.8875675) + x[..., 1]))
            + 0.685028639
            * numpy.sqrt(
                numpy.square((-0.090752731) + x[..., 0]) + numpy.square((-0.409768326) + x[..., 1])
            )
            + 0.495106241
            * numpy.sqrt(
                numpy.square((-0.448765793) + x[..., 0]) + numpy.square((-0.682375923) + x[..., 1])
            )
            + 0.033071629
            * numpy.sqrt(
                numpy.square((-0.462809499) + x[..., 0]) + numpy.square((-0.08615012) + x[..., 1])
            )
            + 0.006303055
            * numpy.sqrt(
                numpy.square((-0.811964203) + x[..., 0]) + numpy.square((-0.570517838) + x[..., 1])
            )
            + 0.364914567
            * numpy.sqrt(
                numpy.square((-0.449999077) + x[..., 0]) + numpy.square((-0.729696493) + x[..., 1])
            )
            + 0.413712133
            * numpy.sqrt(
                numpy.square((-0.954256071) + x[..., 0]) + numpy.square((-0.816834688) + x[..., 1])
            )
            + 0.515864086
            * numpy.sqrt(
                numpy.square((-0.12263947) + x[..., 0]) + numpy.square((-0.909727152) + x[..., 1])
            )
            + 0.468633404
            * numpy.sqrt(
                numpy.square((-0.406598999) + x[..., 0]) + numpy.square((-0.720314503) + x[..., 1])
            )
            + 0.578090911
            * numpy.sqrt(
                numpy.square((-0.886367929) + x[..., 0]) + numpy.square((-0.525692648) + x[..., 1])
            )
            + 0.882884489
            * numpy.sqrt(
                numpy.square((-0.703154798) + x[..., 0]) + numpy.square((-0.505873059) + x[..., 1])
            )
            + 0.243141094
            * numpy.sqrt(
                numpy.square((-0.87491913) + x[..., 0]) + numpy.square((-0.980032187) + x[..., 1])
            )
            + 0.068915527
            * numpy.sqrt(
                numpy.square((-0.555134003) + x[..., 0]) + numpy.square((-0.299554712) + x[..., 1])
            )
            + 0.282246641
            * numpy.sqrt(
                numpy.square((-0.25562684) + x[..., 0]) + numpy.square((-0.027311306) + x[..., 1])
            )
            + 0.105752858
            * numpy.sqrt(
                numpy.square((-0.259177167) + x[..., 0]) + numpy.square((-0.816858348) + x[..., 1])
            )
            + 0.720614408
            * numpy.sqrt(
                numpy.square((-0.35512344) + x[..., 0]) + numpy.square((-0.428500236) + x[..., 1])
            )
            + 0.653215026
            * numpy.sqrt(
                numpy.square((-0.136884117) + x[..., 0]) + numpy.square((-0.436213792) + x[..., 1])
            )
            + 0.624720035
            * numpy.sqrt(
                numpy.square((-0.807065538) + x[..., 0]) + numpy.square((-0.994316024) + x[..., 1])
            )
            + 0.847369399
            * numpy.sqrt(
                numpy.square((-0.325977816) + x[..., 0]) + numpy.square((-0.898069738) + x[..., 1])
            )
            + 0.422595719
            * numpy.sqrt(
                numpy.square((-0.428835591) + x[..., 0]) + numpy.square((-0.633747186) + x[..., 1])
            )
            + 0.944912015
            * numpy.sqrt(
                numpy.square((-0.008958666) + x[..., 0]) + numpy.square((-0.014431655) + x[..., 1])
            )
            + 0.50897735
            * numpy.sqrt(
                numpy.square((-0.224289483) + x[..., 0]) + numpy.square((-0.620459527) + x[..., 1])
            )
            + 0.176196976
            * numpy.sqrt(
                numpy.square((-0.660668676) + x[..., 0]) + numpy.square((-0.974784147) + x[..., 1])
            )
            + 0.316695464
            * numpy.sqrt(
                numpy.square((-0.28740342) + x[..., 0]) + numpy.square((-0.657817639) + x[..., 1])
            )
            + 0.28167686
            * numpy.sqrt(
                numpy.square((-0.131118059) + x[..., 0]) + numpy.square((-0.87756709) + x[..., 1])
            )
            + 0.662635209
            * numpy.sqrt(
                numpy.square((-0.407116195) + x[..., 0]) + numpy.square((-0.287724436) + x[..., 1])
            )
            + 0.57677832
            * numpy.sqrt(
                numpy.square((-0.161623003) + x[..., 0]) + numpy.square((-0.647396747) + x[..., 1])
            )
            + 0.253926782
            * numpy.sqrt(
                numpy.square((-0.861750457) + x[..., 0]) + numpy.square((-0.379377932) + x[..., 1])
            )
            + 0.247462399
            * numpy.sqrt(
                numpy.square((-0.37771215) + x[..., 0]) + numpy.square((-0.53017166) + x[..., 1])
            )
            + 0.591730943
            * numpy.sqrt(
                numpy.square((-0.888609022) + x[..., 0]) + numpy.square((-0.389503757) + x[..., 1])
            )
            + 0.21275857
            * numpy.sqrt(
                numpy.square((-0.269979004) + x[..., 0]) + numpy.square((-0.108049376) + x[..., 1])
            )
            + 0.994995233
            * numpy.sqrt(
                numpy.square((-0.777387678) + x[..., 0]) + numpy.square((-0.556874194) + x[..., 1])
            )
            + 0.033177507
            * numpy.sqrt(
                numpy.square((-0.422785271) + x[..., 0]) + numpy.square((-0.687970277) + x[..., 1])
            )
            + 0.652054839
            * numpy.sqrt(
                numpy.square((-0.429854173) + x[..., 0]) + numpy.square((-0.487995984) + x[..., 1])
            )
            + 0.277760854
            * numpy.sqrt(
                numpy.square((-0.249065869) + x[..., 0]) + numpy.square((-0.071196006) + x[..., 1])
            )
            + 0.624266551
            * numpy.sqrt(
                numpy.square((-0.381769942) + x[..., 0]) + numpy.square((-0.976702526) + x[..., 1])
            )
            + 0.824189458
            * numpy.sqrt(
                numpy.square((-0.070981198) + x[..., 0]) + numpy.square((-0.155248176) + x[..., 1])
            )
            + 0.177354361
            * numpy.sqrt(
                numpy.square((-0.715629012) + x[..., 0]) + numpy.square((-0.967347114) + x[..., 1])
            )
            + 0.253220648
            * numpy.sqrt(
                numpy.square((-0.702904229) + x[..., 0]) + numpy.square((-0.489282642) + x[..., 1])
            )
            + 0.843918263
            * numpy.sqrt(
                numpy.square((-0.070158352) + x[..., 0]) + numpy.square((-0.976876528) + x[..., 1])
            )
            + 0.937565679
            * numpy.sqrt(numpy.square((-0.96850936) + x[..., 0]) + numpy.square((-0.0926592) + x[..., 1]))
            + 0.121318081
            * numpy.sqrt(
                numpy.square((-0.270018449) + x[..., 0]) + numpy.square((-0.273721045) + x[..., 1])
            )
            + 0.343167371
            * numpy.sqrt(
                numpy.square((-0.31814865) + x[..., 0]) + numpy.square((-0.833175308) + x[..., 1])
            )
            + 0.350456488
            * numpy.sqrt(
                numpy.square((-0.883384608) + x[..., 0]) + numpy.square((-0.541837462) + x[..., 1])
            )
            + 0.708073942
            * numpy.sqrt(
                numpy.square((-0.586224197) + x[..., 0]) + numpy.square((-0.550704499) + x[..., 1])
            )
            + 0.991532134
            * numpy.sqrt(
                numpy.square((-0.382088503) + x[..., 0]) + numpy.square((-0.63808083) + x[..., 1])
            )
            + 0.581181944
            * numpy.sqrt(
                numpy.square((-0.972983411) + x[..., 0]) + numpy.square((-0.91715719) + x[..., 1])
            )
            + 0.407455527
            * numpy.sqrt(
                numpy.square((-0.670833182) + x[..., 0]) + numpy.square((-0.54629267) + x[..., 1])
            )
            + 0.634572049
            * numpy.sqrt(
                numpy.square((-0.951145408) + x[..., 0]) + numpy.square((-0.502563846) + x[..., 1])
            )
            + 0.068854317
            * numpy.sqrt(
                numpy.square((-0.718255078) + x[..., 0]) + numpy.square((-0.87386812) + x[..., 1])
            )
            + 0.508948949
            * numpy.sqrt(
                numpy.square((-0.443597148) + x[..., 0]) + numpy.square((-0.021168338) + x[..., 1])
            )
            + 0.974851791
            * numpy.sqrt(
                numpy.square((-0.879729483) + x[..., 0]) + numpy.square((-0.494762232) + x[..., 1])
            )
            + 0.652217096
            * numpy.sqrt(
                numpy.square((-0.469808285) + x[..., 0]) + numpy.square((-0.221419209) + x[..., 1])
            )
            + 0.669588379
            * numpy.sqrt(
                numpy.square((-0.463851194) + x[..., 0]) + numpy.square((-0.80423015) + x[..., 1])
            )
            + 0.499318619
            * numpy.sqrt(
                numpy.square((-0.371433767) + x[..., 0]) + numpy.square((-0.881661683) + x[..., 1])
            )
            + 0.533668985
            * numpy.sqrt(
                numpy.square((-0.118341913) + x[..., 0]) + numpy.square((-0.518248232) + x[..., 1])
            )
            + 0.970272177
            * numpy.sqrt(
                numpy.square((-0.965381458) + x[..., 0]) + numpy.square((-0.097842272) + x[..., 1])
            )
            + 0.889775465
            * numpy.sqrt(
                numpy.square((-0.843545642) + x[..., 0]) + numpy.square((-0.44761195) + x[..., 1])
            )
            + 0.472230298
            * numpy.sqrt(
                numpy.square((-0.721329559) + x[..., 0]) + numpy.square((-0.692572882) + x[..., 1])
            )
            + 0.560106672
            * numpy.sqrt(
                numpy.square((-0.964443768) + x[..., 0]) + numpy.square((-0.061002432) + x[..., 1])
            )
            + 0.692457486
            * numpy.sqrt(
                numpy.square((-0.364572573) + x[..., 0]) + numpy.square((-0.630550079) + x[..., 1])
            )
            + 0.315005085
            * numpy.sqrt(
                numpy.square((-0.768307414) + x[..., 0]) + numpy.square((-0.138185756) + x[..., 1])
            )
            + 0.836435959
            * numpy.sqrt(
                numpy.square((-0.332475646) + x[..., 0]) + numpy.square((-0.240420458) + x[..., 1])
            )
            + 0.120484536
            * numpy.sqrt(
                numpy.square((-0.542078991) + x[..., 0]) + numpy.square((-0.442112552) + x[..., 1])
            )
            + 0.421142313
            * numpy.sqrt(
                numpy.square((-0.138436395) + x[..., 0]) + numpy.square((-0.720072587) + x[..., 1])
            )
            + 0.638195264
            * numpy.sqrt(
                numpy.square((-0.313668031) + x[..., 0]) + numpy.square((-0.520827173) + x[..., 1])
            )
            + 0.625177598
            * numpy.sqrt(
                numpy.square((-0.079880806) + x[..., 0]) + numpy.square((-0.573847499) + x[..., 1])
            )
            + 0.609640636
            * numpy.sqrt(
                numpy.square((-0.934151882) + x[..., 0]) + numpy.square((-0.874003029) + x[..., 1])
            )
            + 0.518604413
            * numpy.sqrt(
                numpy.square((-0.494253899) + x[..., 0]) + numpy.square((-0.87679888) + x[..., 1])
            )
        )
        y12 = (
            0.368396296
            * numpy.sqrt(
                numpy.square((-0.678912408) + x[..., 0]) + numpy.square((-0.237302858) + x[..., 1])
            )
            + 0.345329549
            * numpy.sqrt(
                numpy.square((-0.691564087) + x[..., 0]) + numpy.square((-0.171938211) + x[..., 1])
            )
            + 0.294507034
            * numpy.sqrt(
                numpy.square((-0.054742686) + x[..., 0]) + numpy.square((-0.86924724) + x[..., 1])
            )
            + 0.86520669
            * numpy.sqrt(
                numpy.square((-0.240201887) + x[..., 0]) + numpy.square((-0.226468038) + x[..., 1])
            )
            + 0.87902443
            * numpy.sqrt(
                numpy.square((-0.882748296) + x[..., 0]) + numpy.square((-0.044926617) + x[..., 1])
            )
            + 0.462498595
            * numpy.sqrt(
                numpy.square((-0.100584986) + x[..., 0]) + numpy.square((-0.391464436) + x[..., 1])
            )
            + 0.71823366
            * numpy.sqrt(
                numpy.square((-0.678361309) + x[..., 0]) + numpy.square((-0.491038462) + x[..., 1])
            )
            + 0.14904892
            * numpy.sqrt(
                numpy.square((-0.101378147) + x[..., 0]) + numpy.square((-0.799237045) + x[..., 1])
            )
            + 0.249685483
            * numpy.sqrt(
                numpy.square((-0.042808705) + x[..., 0]) + numpy.square((-0.486127193) + x[..., 1])
            )
            + 0.614732903
            * numpy.sqrt(
                numpy.square((-0.753454355) + x[..., 0]) + numpy.square((-0.547738761) + x[..., 1])
            )
            + 0.781898121
            * numpy.sqrt(
                numpy.square((-0.987038189) + x[..., 0]) + numpy.square((-0.369319268) + x[..., 1])
            )
            + 0.714756462
            * numpy.sqrt(
                numpy.square((-0.014838933) + x[..., 0]) + numpy.square((-0.311926054) + x[..., 1])
            )
            + 0.605330776
            * numpy.sqrt(
                numpy.square((-0.146861446) + x[..., 0]) + numpy.square((-0.234763973) + x[..., 1])
            )
            + 0.560841331
            * numpy.sqrt(
                numpy.square((-0.99097964) + x[..., 0]) + numpy.square((-0.393577238) + x[..., 1])
            )
            + 0.446196383
            * numpy.sqrt(
                numpy.square((-0.132193348) + x[..., 0]) + numpy.square((-0.509515406) + x[..., 1])
            )
            + 0.672457081
            * numpy.sqrt(
                numpy.square((-0.085426504) + x[..., 0]) + numpy.square((-0.099146414) + x[..., 1])
            )
            + 0.098707048
            * numpy.sqrt(
                numpy.square((-0.185948657) + x[..., 0]) + numpy.square((-0.678648932) + x[..., 1])
            )
            + 0.679428127
            * numpy.sqrt(
                numpy.square((-0.790163824) + x[..., 0]) + numpy.square((-0.669306694) + x[..., 1])
            )
            + 0.348148868
            * numpy.sqrt(
                numpy.square((-0.864520008) + x[..., 0]) + numpy.square((-0.278201951) + x[..., 1])
            )
            + 0.156948598
            * numpy.sqrt(
                numpy.square((-0.747931534) + x[..., 0]) + numpy.square((-0.805620061) + x[..., 1])
            )
            + 0.244689225
            * numpy.sqrt(
                numpy.square((-0.389486767) + x[..., 0]) + numpy.square((-0.985045886) + x[..., 1])
            )
            + 0.617963496
            * numpy.sqrt(
                numpy.square((-0.795038324) + x[..., 0]) + numpy.square((-0.406184688) + x[..., 1])
            )
            + 0.459274891
            * numpy.sqrt(
                numpy.square((-0.620982121) + x[..., 0]) + numpy.square((-0.425551683) + x[..., 1])
            )
            + 0.763642228
            * numpy.sqrt(
                numpy.square((-0.761372941) + x[..., 0]) + numpy.square((-0.150512426) + x[..., 1])
            )
            + 0.737356562
            * numpy.sqrt(
                numpy.square((-0.580264759) + x[..., 0]) + numpy.square((-0.57955549) + x[..., 1])
            )
            + 0.375284859
            * numpy.sqrt(
                numpy.square((-0.663559353) + x[..., 0]) + numpy.square((-0.256562025) + x[..., 1])
            )
            + 0.783596499
            * numpy.sqrt(
                numpy.square((-0.082141431) + x[..., 0]) + numpy.square((-0.566966552) + x[..., 1])
            )
            + 0.255581744
            * numpy.sqrt(
                numpy.square((-0.566863651) + x[..., 0]) + numpy.square((-0.767847986) + x[..., 1])
            )
            + 0.775846597
            * numpy.sqrt(
                numpy.square((-0.44322952) + x[..., 0]) + numpy.square((-0.853143121) + x[..., 1])
            )
            + 0.507633257
            * numpy.sqrt(
                numpy.square((-0.328250605) + x[..., 0]) + numpy.square((-0.24228275) + x[..., 1])
            )
            + 0.638106672
            * numpy.sqrt(
                numpy.square((-0.330597357) + x[..., 0]) + numpy.square((-0.523393193) + x[..., 1])
            )
            + 0.524056441
            * numpy.sqrt(
                numpy.square((-0.122911336) + x[..., 0]) + numpy.square((-0.821222482) + x[..., 1])
            )
            + 0.490652718
            * numpy.sqrt(
                numpy.square((-0.617859971) + x[..., 0]) + numpy.square((-0.488656954) + x[..., 1])
            )
            + 0.050514172
            * numpy.sqrt(
                numpy.square((-0.533949065) + x[..., 0]) + numpy.square((-0.110078878) + x[..., 1])
            )
            + 0.89068344
            * numpy.sqrt(
                numpy.square((-0.14407987) + x[..., 0]) + numpy.square((-0.508247822) + x[..., 1])
            )
            + 0.532997258
            * numpy.sqrt(
                numpy.square((-0.392277262) + x[..., 0]) + numpy.square((-0.975798382) + x[..., 1])
            )
            + 0.396906842
            * numpy.sqrt(
                numpy.square((-0.014761557) + x[..., 0]) + numpy.square((-0.997983623) + x[..., 1])
            )
            + 0.964692073
            * numpy.sqrt(
                numpy.square((-0.870770817) + x[..., 0]) + numpy.square((-0.650318083) + x[..., 1])
            )
            + 0.49470031
            * numpy.sqrt(
                numpy.square((-0.245518802) + x[..., 0]) + numpy.square((-0.006101788) + x[..., 1])
            )
            + 0.00108484
            * numpy.sqrt(
                numpy.square((-0.803182518) + x[..., 0]) + numpy.square((-0.6348217) + x[..., 1])
            )
            + 0.292041243
            * numpy.sqrt(
                numpy.square((-0.084030347) + x[..., 0]) + numpy.square((-0.57475456) + x[..., 1])
            )
            + 0.386342913
            * numpy.sqrt(
                numpy.square((-0.987223854) + x[..., 0]) + numpy.square((-0.430708748) + x[..., 1])
            )
            + 0.324370032
            * numpy.sqrt(
                numpy.square((-0.558265263) + x[..., 0]) + numpy.square((-0.716787983) + x[..., 1])
            )
            + 0.793393638
            * numpy.sqrt(
                numpy.square((-0.681922639) + x[..., 0]) + numpy.square((-0.234016315) + x[..., 1])
            )
            + 0.886240758
            * numpy.sqrt(
                numpy.square((-0.859579102) + x[..., 0]) + numpy.square((-0.793757876) + x[..., 1])
            )
            + 0.766127385
            * numpy.sqrt(
                numpy.square((-0.586731618) + x[..., 0]) + numpy.square((-0.772906235) + x[..., 1])
            )
            + 0.689588784
            * numpy.sqrt(
                numpy.square((-0.039519882) + x[..., 0]) + numpy.square((-0.335633815) + x[..., 1])
            )
            + 0.715616656
            * numpy.sqrt(
                numpy.square((-0.912934749) + x[..., 0]) + numpy.square((-0.341198507) + x[..., 1])
            )
            + 0.690676817
            * numpy.sqrt(
                numpy.square((-0.542497487) + x[..., 0]) + numpy.square((-0.089361996) + x[..., 1])
            )
            + 0.742607845
            * numpy.sqrt(
                numpy.square((-0.220985313) + x[..., 0]) + numpy.square((-0.261752873) + x[..., 1])
            )
            + 0.15396918
            * numpy.sqrt(
                numpy.square((-0.925566054) + x[..., 0]) + numpy.square((-0.362064476) + x[..., 1])
            )
            + 0.865736337
            * numpy.sqrt(
                numpy.square((-0.792780726) + x[..., 0]) + numpy.square((-0.228354169) + x[..., 1])
            )
            + 0.567320774
            * numpy.sqrt(
                numpy.square((-0.739380752) + x[..., 0]) + numpy.square((-0.228665243) + x[..., 1])
            )
            + 0.842823231
            * numpy.sqrt(
                numpy.square((-0.677742334) + x[..., 0]) + numpy.square((-0.079001425) + x[..., 1])
            )
            + 0.06996727
            * numpy.sqrt(
                numpy.square((-0.501188372) + x[..., 0]) + numpy.square((-0.86447221) + x[..., 1])
            )
            + 0.467475693
            * numpy.sqrt(
                numpy.square((-0.431123295) + x[..., 0]) + numpy.square((-0.898167826) + x[..., 1])
            )
        )
        return y + y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9 + y10 + y11 + y12