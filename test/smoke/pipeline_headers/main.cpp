int cvh_pipeline_builder_header_compile();
int cvh_pipeline_info_header_compile();
int cvh_pipeline_operations_header_compile();
int cvh_pipeline_pipeline_header_compile();
int cvh_pipeline_plan_header_compile();
int cvh_pipeline_types_header_compile();
int cvh_pipeline_views_header_compile();
int cvh_pipeline_workspace_header_compile();

int main()
{
    return cvh_pipeline_builder_header_compile() +
           cvh_pipeline_info_header_compile() +
           cvh_pipeline_operations_header_compile() +
           cvh_pipeline_pipeline_header_compile() +
           cvh_pipeline_plan_header_compile() +
           cvh_pipeline_types_header_compile() +
           cvh_pipeline_views_header_compile() +
           cvh_pipeline_workspace_header_compile();
}
