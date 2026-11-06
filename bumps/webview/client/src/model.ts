export interface SerializedModel {
  references: { [key: string]: Parameter };
  object: unknown;
  $schema: "bumps-draft-02";
}

export interface NumpyArray {
  values: number[];
  dtype: string;
  __class__: "bumps.util.NumpyArray";
}

export interface Variable {
  value: number;
  __class__: "bumps.parameter.Variable";
}

export interface Parameter {
  id: string;
  name: string;
  fixed: boolean;
  slot: Variable;
  limits: (number | "-inf" | "inf")[];
  bounds: (number | "-inf" | "inf")[] | null;
  tags: string[];
  __class__: "bumps.parameter.Parameter";
}

export interface ParameterLike {
  id: string;
  __class__: "bumps.parameter.Parameter" | "Reference";
}
