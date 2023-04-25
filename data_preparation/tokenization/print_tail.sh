
full_path=/fsx/loubna/code/bigcode-dataset/tokenization/jobs/logs
output_file=/fsx/loubna/data/tokenized_stack_no_pii/code/combined_tail.txt

for file in $full_path/*.out
do
    echo "======================== $file ==========================" >> $output_file
    tail -n30 $file >> $output_file
done