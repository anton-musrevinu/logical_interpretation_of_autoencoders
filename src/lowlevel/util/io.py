

def save_feature_layer(self,model_save_dir, model_save_name, model_idx, feature_layer_ex, feature_layer_hidden_ex = None):
    save_file = os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx)))
    with open(save_file,'w') as f:
        f.write('feature_layer:\n')
        f.write('{}'.format(feature_layer_ex))
       	if feature_layer_hidden_ex != None:
	        f.write('feature_layer hidden representation:\n')
	        f.write('{}'.format(feature_layer_hidden_ex))