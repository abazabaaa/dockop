

# python parse_data.py /data/dockop_data/AmpC_screen_table_clean.feather /data/dockop_data_new/ampc

for fp in morgan; do
    for size in 4096; do
	python main.py $fp $size 100000 /data/newdockop/dockop/code/mod_code_base/logreg_only.json /data/dockop_data_new/ampcprocessed_data
    done
done
