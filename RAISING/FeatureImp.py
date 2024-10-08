def DeepFeatImp(model, feature_list, output_list):
    import numpy
    import pandas
    from tensorflow import keras

    weights = {}
    # 	print(output_list)
    model_layers = [
        layer for layer in model.layers if isinstance(layer, keras.layers.Dense)
    ]
    #  model_layers = model.layers
    #               print(model_layers)
    for layer in model_layers:
        layer_weights = numpy.abs(layer.get_weights()[0])
        weights[layer.name] = layer_weights
    for key, arr in weights.items():
        col_sum = weights[key].sum(axis=0)
        # divide each column by its sum
        weights[key] = arr / col_sum[numpy.newaxis, :]
    # matrix multiplication
    layer_names = [layer.name for layer in model_layers]
    filt_layers = [
        name
        for name in layer_names
        if name.startswith(("multiclass", "continuous", "binary"))
    ]
    dense_layers = [dl for dl in layer_names if dl.startswith("dense")]
    if len(filt_layers) == 0:
        weights_all = list(weights.values())
        feature_concat = numpy.linalg.multi_dot(weights_all)
    else:
        weights_dense = [weights[key] for key in dense_layers]
        if len(weights_dense) > 1:
            results_dense = numpy.linalg.multi_dot(weights_dense)
        else:
            results_dense = weights_dense[0]
        # 		print(results_dense);print(filt_layers)
        feature_cont = []
        for l in filt_layers:
            # results_dense = weights_dense
            feature_cont.append(numpy.linalg.multi_dot([results_dense, weights[l]]))
        # 		print(feature_cont)
        feature_concat = numpy.concatenate(feature_cont, axis=1)
    # 		results_fin = pandas.DataFrame(feature_cont)
    # 	print(feature_concat);print(feature_list)
    feature_list = pandas.DataFrame(feature_list, columns=["features"])
    results_fin = pandas.DataFrame(feature_concat, columns=output_list)
    results_fin = pandas.concat([feature_list, results_fin], axis=1)
    return results_fin
