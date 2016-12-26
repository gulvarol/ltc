This is a directory for datasets in the following format:
```shell
UCF101/  
	annot/    # Contains class indices.
		forceClasses.t7  
	flow/     # Contains flow data in Torch tables of jpgs.
		t7/  
			ApplyEyeMakeup/  
				v_ApplyEyeMakeup_g01_c01.avi.t7  
				...  
			ApplyLipstick/  
			...  
	rgb/     # Contains RGB data in Torch tables of jpgs.
		t7/  
	splits/  # Contains empty files
		split1/  
			train/  
				ApplyEyeMakeup/  
					v_ApplyEyeMakeup_g08_c01.avi  
					...  
			test/  
				ApplyEyeMakeup/  
					v_ApplyEyeMakeup_g01_c01.avi  
					...  
			test_100_4/  
				ApplyEyeMakeup/  
					v_ApplyEyeMakeup_g01_c01.avi_0001.avi  
					...  
		split2/  
		split3/  
HMDB51/     # Same structure as UCF101
```